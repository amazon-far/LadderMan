import datetime
import os
from typing import Sequence, TYPE_CHECKING

import hydra
import imageio
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase as ModBase
from termcolor import colored
from torchrl.envs import ExplorationType, TransformedEnv, set_exploration_type
from torchrl.envs.transforms import Compose, InitTracker, StepCounter, VecNorm
from tqdm import tqdm

import active_adaptation
import active_adaptation.learning  # noqa: F401  (registers the algo configs)
from active_adaptation.utils.checkpoint import load_checkpoint
from active_adaptation.utils.wandb import parse_checkpoint_path, resolve_local_path

if TYPE_CHECKING:
    from active_adaptation.envs.base import _Env


def load_state(path: str | None, project: str | None = None) -> dict:
    path = parse_checkpoint_path(path, project=project)
    if path is None:
        return {}
    print(colored(f"[Info]: Loading {path}", "green"))
    return load_checkpoint(path)


def make_env_policy(cfg: DictConfig):
    """Build the environment (with observation normalization) and the algorithm.

    - ``cfg.vecnorm_path``: file holding the observation-normalization statistics (a checkpoint or
      a stats file such as ``data/vecnorm/ladder_vecnorm.pt``). Takes precedence over the stats
      stored in ``cfg.checkpoint_path``.
    - ``cfg.vecnorm``: ``"eval"`` keeps the statistics frozen, ``"train"`` keeps updating them.
    - ``cfg.checkpoint_path``: initialize / resume the policy from this checkpoint.
    """
    OmegaConf.set_struct(cfg, False)
    from active_adaptation.envs import LadderEnv

    project = cfg.wandb.get("project", None)
    # teachers of the ME-DAgger student: resolve (and download if needed) once, before building the policy
    if cfg.algo.get("teacher_checkpoint_path_list", None):
        cfg.algo.teacher_checkpoint_path_list = [
            parse_checkpoint_path(p, project=project) for p in cfg.algo.teacher_checkpoint_path_list
        ]

    # only keep the observation groups consumed by the algorithm (+ groups ending with "_")
    policy_in_keys = cfg.algo.get("in_keys", ["policy", "priv"])
    for obs_group_key in list(cfg.task.observation.keys()):
        if obs_group_key not in policy_in_keys and not obs_group_key.endswith("_"):
            cfg.task.observation.pop(obs_group_key)
            print(colored(f"Discard obs group {obs_group_key} as it is not used.", "yellow"))

    base_env = LadderEnv(cfg.task)

    state_dict = load_state(cfg.get("checkpoint_path", None), project)

    obs_keys = [
        key for key, spec in base_env.observation_spec.items(True, True)
        if not (spec.dtype == bool or key.endswith("_"))
    ]
    transform = Compose(InitTracker(), StepCounter())
    assert cfg.vecnorm in ("train", "eval", None)
    print(colored(f"[Info]: create VecNorm for keys: {obs_keys}", "green"))
    vecnorm = VecNorm(obs_keys, decay=0.9999)
    vecnorm(base_env.fake_tensordict())

    vecnorm_path = cfg.get("vecnorm_path", None)
    if vecnorm_path is not None:
        vecnorm_state = load_state(resolve_local_path(vecnorm_path))
        vecnorm.load_state_dict(vecnorm_state.get("vecnorm", vecnorm_state))
        print(colored(f"[Info]: Load VecNorm from {vecnorm_path}.", "green"))
    elif "vecnorm" in state_dict:
        vecnorm.load_state_dict(state_dict["vecnorm"])
        print(colored("[Info]: Load VecNorm from checkpoint.", "green"))
    if cfg.vecnorm == "train":
        print(colored("[Info]: Updating observation normalizer.", "green"))
        transform.append(vecnorm)
    elif cfg.vecnorm == "eval":
        print(colored("[Info]: Not updating observation normalizer.", "green"))
        transform.append(vecnorm.to_observation_norm())

    env = TransformedEnv(base_env, transform)
    env.set_seed(cfg.seed)

    policy_cls = hydra.utils.get_class(cfg.algo._target_)
    active_adaptation.print(f"Creating policy {policy_cls} on device {base_env.device}")
    policy = policy_cls(
        cfg.algo,
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device=base_env.device,
        env=env,
    )
    if "policy" in state_dict:
        print(colored("[Info]: Load policy from checkpoint.", "green"))
        policy.load_state_dict(state_dict["policy"])

    env: _Env
    return env, policy, vecnorm


class ObsNorm(ModBase):
    """Frozen observation normalization, prepended to the exported policy."""

    def __init__(self, in_keys, out_keys, locs, scales):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys
        self.loc = nn.ParameterDict({k: nn.Parameter(locs[k]) for k in in_keys})
        self.scale = nn.ParameterDict({k: nn.Parameter(scales[k]) for k in out_keys})
        self.requires_grad_(False)

    def forward(self, tensordict: TensorDictBase):
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            obs = tensordict.get(in_key, None)
            if obs is not None:
                tensordict.set(out_key, (obs - self.loc[in_key]) / self.scale[out_key])
        return tensordict

    @classmethod
    def from_vecnorm(cls, vecnorm: VecNorm, keys):
        in_keys, out_keys = [], []
        for in_key, out_key in zip(vecnorm.in_keys, vecnorm.out_keys):
            if in_key in keys:
                in_keys.append(in_key)
                out_keys.append(out_key)
        return cls(in_keys=in_keys, out_keys=out_keys, locs=vecnorm.loc, scales=vecnorm.scale)


class ObsOODDetector(ModBase):
    """Reports the fraction of normalized observations beyond ``sigma`` (useful on hardware)."""

    def __init__(self, in_keys, sigma=5.0):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = [("next", f"{k}_ood_ratio") for k in in_keys] + [("next", k) for k in in_keys]
        self.sigma = sigma

    def forward(self, tensordict: TensorDictBase):
        for in_key in self.in_keys:
            obs = tensordict.get(in_key, None)
            if obs is not None:
                ood_ratio = (obs.abs() > self.sigma).float().mean(dim=tuple(range(1, obs.ndim)))
                tensordict.set(("next", f"{in_key}_ood_ratio"), ood_ratio)
                tensordict.set(("next", in_key), obs)
        return tensordict


class EpisodeStats:
    def __init__(self, in_keys: Sequence[str], device: torch.device):
        self.in_keys = in_keys
        self.device = device
        self._stats = TensorDict({key: torch.tensor([0.], device=device) for key in in_keys}, [1])
        self._episodes = torch.tensor(0, device=device)

    def add(self, tensordict: TensorDictBase) -> int:
        next_tensordict = tensordict["next"]
        done = next_tensordict["done"]
        if done.any():
            done = done.squeeze(-1)
            next_tensordict = next_tensordict.select(*self.in_keys)
            self._stats = self._stats + next_tensordict[done].sum(dim=0)
            self._episodes += done.sum()
        return len(self)

    def pop(self):
        stats = self._stats / self._episodes
        self._stats.zero_()
        self._episodes.zero_()
        return stats.cpu()

    def __len__(self):
        return self._episodes.item()


@torch.inference_mode()
def evaluate(env: TransformedEnv, policy: torch.nn.Module, seed: int = 0, render: bool = False):
    """Roll out one episode per env with the deterministic policy and report episode statistics."""
    env.base_env.eval()
    env.eval()
    env.set_seed(seed)

    tensordict_ = env.reset()
    trajs, frames = [], []
    with set_exploration_type(ExplorationType.MODE):
        for _ in tqdm(range(env.max_episode_length), miniters=10):
            tensordict_ = policy(tensordict_)
            tensordict, tensordict_ = env.step_and_maybe_reset(tensordict_)
            trajs.append(tensordict.select(("next", "done"), ("next", "stats"), strict=False).cpu())
            if render:
                frames.append(env.render(mode="rgb_array"))

    trajs: TensorDictBase = torch.stack(trajs, dim=1)
    done = trajs.get(("next", "done"))
    first_done = torch.argmax(done.long(), dim=1).cpu()

    def take_first_episode(tensor: torch.Tensor):
        indices = first_done.reshape(first_done.shape + (1,) * (tensor.ndim - 2))
        return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

    info = {}
    episode_len = take_first_episode(trajs["next", "stats", "episode_len"])
    for k, v in trajs["next", "stats"].items(True, True):
        v = take_first_episode(v)
        if k not in ("episode_len", "success"):
            v = v.float() / episode_len.float()
        key = "eval/" + ("/".join(k) if isinstance(k, tuple) else k)
        info[key] = torch.mean(v.float()).item()
        info[key + "_std"] = torch.std(v.float()).item()

    if frames:
        time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
        video_path = os.path.join(os.getcwd(), f"recording-{time_str}.mp4")
        imageio.mimwrite(video_path, np.stack(frames), fps=int(1 / env.step_dt), codec="libx264")
        print(f"Saved video to {video_path}")

    info["episode_cnt"] = len(done.nonzero())
    return dict(sorted(info.items()))
