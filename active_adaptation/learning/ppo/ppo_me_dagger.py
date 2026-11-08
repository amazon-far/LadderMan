"""Step 2: depth student trained with multi-expert DAgger and PPO in a single run.

One student policy (depth CNN + MLP) is trained from scratch with

    L = lambda(t) * bc_coef * L_bc + (1 - lambda(t)) * (L_ppo + L_entropy) + L_value

* ``L_bc``: DAgger behaviour cloning. Every state visited by the student is labelled
  by the teacher whose ladder (inclination, then rung spacing) is closest to the env's ladder
  (multi-expert), and the student mean is regressed onto the teacher mean.
* ``L_ppo`` / ``L_value``: on-policy PPO on the task reward with a privileged critic.
* ``lambda(t)`` (default schedule, total 13000 iterations):
    - ``t < bc_hold_iters`` (7500):           lambda = 1, pure multi-expert DAgger
    - next ``bc_decay_iters`` (2500):        lambda decays linearly 1 -> 0, BC and PPO mixed
    - afterwards (3000):                     lambda = 0, pure PPO; the teachers are released
* While lambda = 1 the rollouts use the deterministic student action (as in DAgger) and the
  BC update runs ``bc_epochs`` epochs; otherwise actions are sampled and ``ppo_epochs`` are used.
* Critic: with ``critic_warmup`` the critic is trained from the start (value loss weight 1) while
  its gradient into the shared depth encoder is scaled by ``1 - lambda``, so the pure-DAgger
  phase updates the policy exactly like BC alone. Without it, the value loss is scaled by
  ``1 - lambda`` (the critic starts learning together with PPO).
* ``bc_phase_strict_position``: the ladder-contact termination is relaxed (``strict_position=False``)
  while lambda = 1, as in the original DAgger stage, and uses the task setting once PPO starts.
* Learning rate: constant ``lr`` while ``lambda > 0`` (BC moves the policy a lot, which
  would otherwise collapse a KL-adaptive learning rate); KL-adaptive afterwards.

Networks (state-dict keys are compatible with ME-DAgger / RL-finetuned student checkpoints):
    dagger_depth_encoder: depth                                      -> dagger_depth_feat
    dagger_policy:        [command, policy, dagger_depth_feat]       -> action (absolute)
    critic:               [command, dagger_depth_feat, object, policy, priv] -> value
"""

from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.utils._pytree as pytree
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential as Seq
from torchrl.data import CompositeSpec, TensorSpec
from torchrl.envs.utils import ExplorationType

from ..modules.distributions import IndependentNormal
from ..utils.valuenorm import ValueNorm1, ValueNormFake
from .common import (
    ACTION_KEY, CMD_KEY, DEPTH_KEY, DONE_KEY, OBJECT_KEY, OBS_KEY, OBS_PRIV_KEY, REWARD_KEY, TERM_KEY,
    GAE, actor_std, build_actor, build_critic, build_depth_encoder, make_batch, orthogonal_init_,
)
from .ppo_teacher import TeacherConfig, build_teacher_networks

torch.set_float32_matmul_precision("high")

DEPTH_FEAT_KEY = "dagger_depth_feat"
EXPERT_ACTION_KEY = "expert_action"


@dataclass
class MEDaggerConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_me_dagger.PPOMEDagger"
    name: str = "ppo_me_dagger"
    max_iters: int = -1  # number of training iterations (<= 0: use `total_frames`)

    # teachers: one checkpoint per expert. By default the experts are the env's grid
    # `inclination_deg_list x spacing_list` of `ladder_inclination_me_randomization`, in
    # inclination-major order. Set the two lists below to give each teacher's (inclination, spacing)
    # explicitly instead (same order as the checkpoints).
    teacher_checkpoint_path_list: List[str] = field(default_factory=list)
    teacher_inclination_deg_list: List[float] = field(default_factory=list)
    teacher_spacing_list: List[float] = field(default_factory=list)
    # architecture of the teachers (must match how they were trained)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)

    # schedule: lambda = 1 for bc_hold_iters, then linear decay to 0 over bc_decay_iters, then 0
    bc_coef: float = 1.0                  # scale of the BC loss
    bc_hold_iters: int = 7500
    bc_decay_iters: int = 2500
    bc_epochs: int = 4                    # epochs per iteration while lambda = 1 (pure DAgger)
    deterministic_bc_rollout: bool = True # roll out the student mean while lambda = 1 (pure DAgger)
    critic_warmup: bool = True            # train the critic during the BC phase (see module docstring)
    # value of the ladder termination's `strict_position` while lambda = 1; once lambda < 1 the
    # value from the task config is restored. None: never change it.
    bc_phase_strict_position: bool | None = False

    # PPO
    train_every: int = 32
    ppo_epochs: int = 3
    num_minibatches: int = 8
    clip_param: float = 0.2
    gamma: float = 0.99
    lmbda: float = 0.95
    lr: float = 3e-4
    desired_kl: float | None = 0.01  # only used once the BC term has been annealed to 0
    entropy_coef_start: float = 0.001
    entropy_coef_end: float = 0.001
    entropy_decay_iters: int = 1000
    init_noise_scale: float = 0.5
    load_noise_scale: float | None = 0.5
    max_grad_norm: float = 1.0
    value_norm: bool = False
    normalize_before_sum: bool = False
    clip_neg_reward: bool = False
    normalize_ratio: bool = False

    vecnorm: str | None = "eval"
    in_keys: List[str] = field(default_factory=lambda: [CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY, DEPTH_KEY])


cs = ConfigStore.instance()
cs.store("ppo_me_dagger", node=MEDaggerConfig(), group="algo")


class MultiExpertLabeler(TensorDictModuleBase):
    """Route each env to one teacher and write its action mean to ``expert_action``.

    Routing is hierarchical: the closest ladder inclination first, then, among the teachers with
    that inclination, the closest rung spacing. The per-env inclination / spacing are read from
    ``env.ladder_inclination_rad`` / ``env.ladder_spacing`` (set at reset by the
    ``ladder_inclination_me_randomization`` randomization).
    """

    def __init__(self, env, teacher_encoders: nn.ModuleList, teacher_actors: nn.ModuleList,
                 teacher_inclination_deg_list: Sequence[float], teacher_spacing_list: Sequence[float],
                 action_dim: int):
        super().__init__()
        self.in_keys = []
        self.out_keys = [EXPERT_ACTION_KEY]
        object.__setattr__(self, "_env", env)
        self.teacher_encoders = teacher_encoders
        self.teacher_actors = teacher_actors
        self.action_dim = action_dim
        self.register_buffer(
            "_teacher_incl_deg", torch.tensor(list(teacher_inclination_deg_list), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "_teacher_spacing", torch.tensor(list(teacher_spacing_list), dtype=torch.float32), persistent=False
        )
        self.active = True

    def release(self):
        """Drop the teachers (called once the BC weight has been annealed to 0)."""
        self.active = False
        self.teacher_encoders = nn.ModuleList()
        self.teacher_actors = nn.ModuleList()

    @torch.no_grad()
    def forward(self, tensordict: TensorDict) -> TensorDict:
        if not self.active:
            # keep the key so that the rollout buffer layout does not change
            tensordict.set(EXPERT_ACTION_KEY, torch.zeros(*tensordict.batch_size, self.action_dim, device=tensordict.device))
            return tensordict
        incl_rad, spacing = self._env.ladder_inclination_rad, self._env.ladder_spacing
        assert incl_rad.ndim == 2 and incl_rad.shape[1] == 1, f"expected [N, 1], got {incl_rad.shape}"
        assert spacing.ndim == 2 and spacing.shape[1] == 1, f"expected [N, 1], got {spacing.shape}"
        incl_deg = incl_rad.squeeze(-1) * (180.0 / math.pi)  # [N]
        spacing = spacing.squeeze(-1)                        # [N]

        # nearest inclination first, then nearest spacing among the teachers tied on inclination
        dist_incl = (incl_deg.unsqueeze(-1) - self._teacher_incl_deg.to(incl_deg.device).unsqueeze(0)).abs()  # [N, E]
        incl_tie = dist_incl <= dist_incl.min(dim=-1, keepdim=True).values + 1e-4
        dist_spacing = (spacing.unsqueeze(-1) - self._teacher_spacing.to(spacing.device).unsqueeze(0)).abs()  # [N, E]
        expert_idx = dist_spacing.masked_fill(~incl_tie, float("inf")).argmin(dim=-1)  # [N]

        locs = []
        for encoder, actor in zip(self.teacher_encoders, self.teacher_actors):
            encoder(tensordict)
            locs.append(actor.get_dist(tensordict).mean.clone())
        locs = torch.stack(locs, dim=0)  # [E, N, act_dim]
        env_ids = torch.arange(locs.shape[1], device=locs.device)
        tensordict.set(EXPERT_ACTION_KEY, locs[expert_idx, env_ids])
        return tensordict


class PPOMEDagger(TensorDictModuleBase):
    def __init__(
        self,
        cfg: MEDaggerConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device,
        env,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.observation_spec = observation_spec
        object.__setattr__(self, "env", env)
        assert observation_spec.get(DEPTH_KEY, None) is not None, "the student requires the `depth` observation group"

        self.entropy_coef = cfg.entropy_coef_start
        self.desired_kl = cfg.desired_kl
        self.clip_param = cfg.clip_param
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.bc_loss_fn = nn.MSELoss(reduction="none")
        self.gae = GAE(gamma=cfg.gamma, lmbda=cfg.lmbda)
        self.reward_groups = list(env.cfg.reward.keys())
        num_reward_groups = len(self.reward_groups)
        self.reward_scales = torch.ones(num_reward_groups, device=self.device) / num_reward_groups
        self.value_norm = (ValueNorm1 if cfg.value_norm else ValueNormFake)(input_shape=num_reward_groups).to(self.device)

        self.action_dim = action_spec.shape[-1]
        self.joint_names = env.action_manager.joint_names
        self.dist_cls = IndependentNormal
        self.dist_keys = IndependentNormal.dist_keys

        # ---- student ----
        self.dagger_depth_encoder = build_depth_encoder(DEPTH_FEAT_KEY, device)
        self.dagger_policy = build_actor(
            [CMD_KEY, OBS_KEY, DEPTH_FEAT_KEY], self.action_dim,
            init_noise_scale=cfg.init_noise_scale, load_noise_scale=cfg.load_noise_scale, device=device,
        )
        critic_in_keys = [CMD_KEY, OBS_KEY, OBS_PRIV_KEY, DEPTH_FEAT_KEY]
        if observation_spec.get(OBJECT_KEY, None) is not None:
            critic_in_keys.append(OBJECT_KEY)
        self.critic = build_critic(critic_in_keys, num_reward_groups, device)

        fake_input = observation_spec.zero()
        self.dagger_depth_encoder(fake_input)
        self.dagger_policy(fake_input)
        self.critic(fake_input)
        self.apply(orthogonal_init_)

        self._task_strict_position = None  # see `begin_iteration`
        # teachers are loaded lazily by `get_rollout_policy("train")`, i.e. only when training
        self.labeler: MultiExpertLabeler | None = None

        self.lr_policy = cfg.lr
        self.opt_policy = torch.optim.Adam(
            [{"params": self.dagger_policy.parameters()}, {"params": self.dagger_depth_encoder.parameters()}],
            lr=self.lr_policy,
        )
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.num_updates = 0

    def _load_teachers(self) -> MultiExpertLabeler:
        from active_adaptation.utils.checkpoint import load_checkpoint
        from active_adaptation.utils.wandb import parse_checkpoint_path

        cfg = self.cfg
        num_teachers = len(cfg.teacher_checkpoint_path_list)
        if num_teachers == 0:
            raise ValueError("ME-DAgger needs teachers: set algo.teacher_checkpoint_path_list.")
        incl_list, spacing_list = list(cfg.teacher_inclination_deg_list), list(cfg.teacher_spacing_list)
        experts = getattr(self.env, "ladder_me_experts", None)
        if not incl_list or not spacing_list:
            if experts is None:
                raise ValueError("Set algo.teacher_inclination_deg_list / teacher_spacing_list, or enable "
                                 "the `ladder_inclination_me_randomization` randomization.")
            incl_list = incl_list or [e[0] for e in experts]
            spacing_list = spacing_list or [e[1] for e in experts]
        if not (num_teachers == len(incl_list) == len(spacing_list)):
            raise ValueError(
                f"Got {num_teachers} teacher checkpoints for {len(incl_list)} inclinations / {len(spacing_list)} spacings. "
                f"Expert grid of the env (inclination-major order): {experts}"
            )
        if experts is not None and set(zip(incl_list, spacing_list)) != set(experts):
            warnings.warn(f"Teachers {list(zip(incl_list, spacing_list))} do not match the env's expert grid {experts}.")
        fake_input = self.observation_spec.zero()
        encoders, actors = nn.ModuleList(), nn.ModuleList()
        for path, incl, sp in zip(cfg.teacher_checkpoint_path_list, incl_list, spacing_list):
            local_path = parse_checkpoint_path(path)
            policy_sd = load_checkpoint(local_path, map_location=self.device)["policy"]
            encoder, actor = build_teacher_networks(cfg.teacher, self.observation_spec, self.action_dim, self.device)
            encoder(fake_input)
            actor(fake_input)
            encoder.load_state_dict(policy_sd["encoder_priv"])
            actor.load_state_dict(policy_sd["actor"])
            encoders.append(encoder.requires_grad_(False).eval())
            actors.append(actor.requires_grad_(False).eval())
            print(f"[ME-DAgger] loaded teacher for inclination {incl} deg / rung spacing {sp} m from {local_path}")
        return MultiExpertLabeler(self.env, encoders, actors, incl_list, spacing_list, self.action_dim)

    # ------------------------------------------------------------------
    def bc_weight(self, it: int) -> float:
        """lambda(t) in [0, 1]: weight of BC; PPO is weighted by 1 - lambda."""
        cfg = self.cfg
        if it < cfg.bc_hold_iters:
            return 1.0
        if cfg.bc_decay_iters <= 0:
            return 0.0
        progress = (it - cfg.bc_hold_iters) / cfg.bc_decay_iters
        return float(max(0.0, 1.0 - progress))

    def begin_iteration(self):
        """Per-iteration hook (called by scripts/train.py after the iteration counter is updated)."""
        if self.cfg.bc_phase_strict_position is None:
            return
        term = self.env.termination_funcs.get("cum_ladder_lost_contact_steps", None)
        if term is None:
            return
        if self._task_strict_position is None:
            self._task_strict_position = term.strict_position
        if self.bc_weight(self.env.current_iter) >= 1.0:
            term.strict_position = bool(self.cfg.bc_phase_strict_position)
        else:
            term.strict_position = self._task_strict_position

    def exploration_type(self) -> ExplorationType:
        """Rollout exploration for the current iteration (queried by scripts/train.py)."""
        if self.cfg.deterministic_bc_rollout and self.bc_weight(self.env.current_iter) >= 1.0:
            return ExplorationType.MODE
        return ExplorationType.RANDOM

    def get_rollout_policy(self, mode: str = "train"):
        modules = [self.dagger_depth_encoder, self.dagger_policy]
        out_keys = ["sample_log_prob", "action"] + self.dist_keys + [DEPTH_FEAT_KEY]
        if mode == "train":
            if self.labeler is None:
                if self.bc_weight(self.env.current_iter) > 0:
                    self.labeler = self._load_teachers()
                else:
                    # BC already finished (e.g. resumed late): `expert_action` is only a placeholder
                    self.labeler = MultiExpertLabeler(self.env, nn.ModuleList(), nn.ModuleList(), [], [], self.action_dim)
                    self.labeler.release()
            modules = [self.labeler] + modules
            out_keys = out_keys + [EXPERT_ACTION_KEY]
        return Seq(*modules, selected_out_keys=out_keys)

    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.exclude("stats")
        bc_weight = self.bc_weight(self.env.current_iter)
        info = self.train_policy(tensordict.copy(), bc_weight)
        if bc_weight == 0.0 and self.labeler is not None and self.labeler.active:
            self.labeler.release()
            torch.cuda.empty_cache()
            print(f"[ME-DAgger] BC weight reached 0 at iteration {self.env.current_iter}: teachers released, pure PPO from now on.")
        self.num_updates += 1
        std = actor_std(self.dagger_policy).detach()
        for joint_name, s in zip(self.joint_names, std):
            info[f"actor_std/{joint_name}"] = s
        info["actor_std/mean"] = std.mean()
        info["dagger/bc_weight"] = bc_weight
        info["dagger/ppo_weight"] = 1.0 - bc_weight
        term = self.env.termination_funcs.get("cum_ladder_lost_contact_steps", None)
        if term is not None:
            info["dagger/strict_position"] = float(term.strict_position)
        return info

    def train_policy(self, tensordict: TensorDict, bc_weight: float):
        infos = []
        self._compute_advantage(tensordict, self.critic, "adv", "ret", update_value_norm=True)

        entropy_progress = float(np.clip(self.env.current_iter / self.cfg.entropy_decay_iters, 0.0, 1.0))
        self.entropy_coef = self.cfg.entropy_coef_start + (self.cfg.entropy_coef_end - self.cfg.entropy_coef_start) * entropy_progress

        adaptive_lr = self.desired_kl is not None and bc_weight == 0.0
        if not adaptive_lr:
            self.lr_policy = self.cfg.lr
            for param_group in self.opt_policy.param_groups:
                param_group["lr"] = self.lr_policy

        num_epochs = self.cfg.bc_epochs if bc_weight >= 1.0 else self.cfg.ppo_epochs
        for _epoch in range(num_epochs):
            for minibatch in make_batch(tensordict, self.cfg.num_minibatches):
                infos.append(self._update(minibatch, bc_weight))
                if adaptive_lr:
                    kl = infos[-1]["actor/kl"]
                    if kl > self.desired_kl * 2.0:
                        self.lr_policy = max(1e-5, self.lr_policy / 1.5)
                    elif kl < self.desired_kl / 2.0 and kl > 0.0:
                        self.lr_policy = min(1e-2, self.lr_policy * 1.5)
                    for param_group in self.opt_policy.param_groups:
                        param_group["lr"] = self.lr_policy

        infos = pytree.tree_map(lambda *xs: sum(xs).item() / len(xs), *infos)
        infos["actor/lr"] = self.lr_policy
        infos["actor/entropy_coef"] = self.entropy_coef
        ret = tensordict["ret"]
        ret_mean, ret_std = ret.mean(dim=(0, 1)), ret.std(dim=(0, 1))
        for i, group_name in enumerate(self.reward_groups):
            infos[f"critic/{group_name}.ret_mean"] = ret_mean[i].item()
            infos[f"critic/{group_name}.ret_std"] = ret_std[i].item()
            infos[f"critic/{group_name}.neg_rew_ratio"] = (tensordict[REWARD_KEY][:, :, i] <= 0.0).float().mean().item()
        return dict(sorted(infos.items()))

    @torch.no_grad()
    def _compute_advantage(self, tensordict: TensorDict, critic, adv_key="adv", ret_key="ret", update_value_norm=True):
        keys = tensordict.keys(True, True)
        if not ("state_value" in keys and ("next", "state_value") in keys):
            with tensordict.view(-1) as tensordict_flat:
                self.dagger_depth_encoder(tensordict_flat)
                critic(tensordict_flat)
                next_td = tensordict_flat["next"]
                self.dagger_depth_encoder(next_td)
                critic(next_td)

        values = self.value_norm.denormalize(tensordict["state_value"])
        next_values = self.value_norm.denormalize(tensordict["next", "state_value"])
        rewards = tensordict[REWARD_KEY]
        if self.cfg.clip_neg_reward:
            rewards = rewards.clamp_min(0.0)
        adv, ret = self.gae(rewards, tensordict[TERM_KEY], tensordict[DONE_KEY], values, next_values, tensordict["next", "discount"])

        if self.cfg.normalize_before_sum:
            adv_norm = (adv - adv.mean(dim=(0, 1))) / (adv.std(dim=(0, 1)) + 0.01)
            adv_final = (adv_norm * self.reward_scales).sum(dim=2, keepdim=True)
        else:
            adv_sum = (adv * self.reward_scales).sum(dim=2, keepdim=True)
            adv_final = (adv_sum - adv_sum.mean(dim=(0, 1))) / (adv_sum.std(dim=(0, 1)) + 1e-8)

        if update_value_norm:
            self.value_norm.update(ret)
        tensordict.set(adv_key, adv_final)
        tensordict.set(ret_key, self.value_norm.normalize(ret))
        return tensordict

    def _update(self, tensordict: TensorDict, bc_weight: float):
        dist_kwargs_old = tensordict.select(*self.dist_keys)
        self.dagger_depth_encoder(tensordict)
        dist: D.Independent = self.dagger_policy.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy().mean()
        valid = (tensordict["step_count"] > 5).squeeze(-1)

        # PPO
        adv = tensordict["adv"]
        log_ratio = (log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        ratio = torch.exp(log_ratio)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
        if self.cfg.normalize_ratio:
            clamped_ratio = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param).detach()
            surr1, surr2 = surr1 / clamped_ratio, surr2 / clamped_ratio
        policy_loss = -(torch.min(surr1, surr2)[valid]).mean()
        entropy_loss = -self.entropy_coef * entropy

        ppo_weight = 1.0 - bc_weight
        if self.cfg.critic_warmup:
            # value loss always on, but its gradient into the shared depth encoder is scaled by (1 - lambda)
            if ppo_weight < 1.0:
                feat = tensordict[DEPTH_FEAT_KEY]
                tensordict.set(DEPTH_FEAT_KEY, ppo_weight * feat + (1.0 - ppo_weight) * feat.detach())
            value_weight = 1.0
        else:
            value_weight = ppo_weight
        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)[valid].mean(dim=0)

        loss = ppo_weight * (policy_loss + entropy_loss) + value_weight * value_loss.mean()

        # multi-expert DAgger (behaviour cloning on the student mean)
        if bc_weight > 0.0:
            bc_per_sample = self.bc_loss_fn(dist.mean, tensordict[EXPERT_ACTION_KEY]).mean(dim=-1)
            bc_loss = bc_per_sample[valid].mean()
            loss = loss + bc_weight * self.cfg.bc_coef * bc_loss
        else:
            bc_loss = torch.zeros((), device=self.device)

        self.opt_policy.zero_grad()
        self.opt_critic.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.dagger_policy.parameters(), self.cfg.max_grad_norm)
        depth_grad_norm = nn.utils.clip_grad_norm_(self.dagger_depth_encoder.parameters(), self.cfg.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.opt_policy.step()
        self.opt_critic.step()

        with torch.no_grad():
            explained_var = 1 - value_loss / b_returns[valid].var(dim=0)
            clipfrac = ((ratio - 1.0).abs() > self.clip_param).float().mean()
            kl = D.kl_divergence(self.dist_cls(**dist_kwargs_old), dist).mean()

        info = {
            "actor/policy_loss": policy_loss.detach(),
            "actor/entropy": entropy.detach(),
            "actor/mean_std": tensordict["scale"].detach().mean(),
            "actor/grad_norm": actor_grad_norm,
            "actor/depth_grad_norm": depth_grad_norm,
            "actor/clamp_ratio": clipfrac,
            "actor/kl": kl,
            "actor/approx_kl": ((ratio - 1) - log_ratio).mean(),
            "critic/grad_norm": critic_grad_norm,
            "dagger/bc_loss": bc_loss.detach(),
        }
        for i, group_name in enumerate(self.reward_groups):
            info[f"critic/{group_name}.explained_var"] = explained_var[i]
            info[f"critic/{group_name}.value_loss"] = value_loss[i].detach()
        return info

    # ------------------------------------------------------------------
    def state_dict(self):
        od = OrderedDict()
        for name in ("dagger_depth_encoder", "dagger_policy", "critic", "value_norm"):
            od[name] = getattr(self, name).state_dict()
        od["last_iter"] = self.env.current_iter
        od["lr_policy"] = self.lr_policy
        return od

    def load_state_dict(self, state_dict, strict=True):
        """Loads a student checkpoint (this algorithm, or the older ME-DAgger / finetune checkpoints).

        The critic of an ME-DAgger checkpoint has a different input and is skipped with a warning.
        """
        failed = []
        for name in ("dagger_depth_encoder", "dagger_policy", "critic", "value_norm"):
            if name not in state_dict:
                failed.append(name)
                continue
            try:
                getattr(self, name).load_state_dict(state_dict[name], strict=strict)
            except Exception as e:
                warnings.warn(f"Skipping {name} (shape mismatch or missing keys): {e}")
                failed.append(name)
        if "dagger_policy" in failed or "dagger_depth_encoder" in failed:
            raise RuntimeError(f"Could not load the student policy from the checkpoint (failed: {failed}).")
        self.env.set_progress(state_dict.get("last_iter", 0))
        lr_policy = state_dict.get("lr_policy", None)
        if lr_policy is not None:
            self.lr_policy = lr_policy
            for param_group in self.opt_policy.param_groups:
                param_group["lr"] = self.lr_policy
        return failed
