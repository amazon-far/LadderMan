"""Train the teacher (step 1) or the depth student with joint ME-DAgger + PPO (step 2).

Examples:
    python scripts/train.py algo=teacher task=ladder/teacher task.command.ladder_config.inclination_deg=65.5
    python scripts/train.py algo=student task=ladder/student algo.teacher_checkpoint_path_list=[...]
"""
import datetime
import logging
import os
import subprocess
import sys
import time
from collections import OrderedDict

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from setproctitle import setproctitle
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tqdm import tqdm

from isaaclab.app import AppLauncher

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(FILE_PATH, ".."))
CONFIG_PATH = os.path.join(REPO_ROOT, "cfg")
sys.path.insert(0, REPO_ROOT)  # so that `scripts.helpers` is importable when run as `python scripts/train.py`

import active_adaptation  # noqa: E402,F401  (registers the algorithm configs with Hydra)


# metrics printed to the console every `log_interval` iterations (everything is logged to W&B)
PRINT_KEYS = ["train/stats/episode_len", "train/stats/success", "rollout_fps", "actor/lr", "actor/kl", "dagger/bc_weight", "dagger/ppo_weight", "dagger/bc_loss", "actor_std/mean", "dagger/strict_position"]


def _get_git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return "unknown"


@hydra.main(config_path=CONFIG_PATH, config_name="train", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app))
    simulation_app = app_launcher.app

    # imported after the simulator is launched
    from scripts.helpers import EpisodeStats, evaluate, make_env_policy

    output_dir = os.getcwd()  # Hydra run directory (outputs/<date>/<time>-<task>-<algo>)
    run_name = f"{cfg.exp_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        mode=cfg.wandb.mode,
        tags=cfg.wandb.tags,
        dir=output_dir,
        config=OmegaConf.to_container(cfg),
    )
    git_commit = _get_git_commit()
    run.config.update({"git_commit": git_commit}, allow_val_change=True)
    setproctitle(run_name)
    OmegaConf.save(cfg, os.path.join(output_dir, "cfg.yaml"))

    env, policy, vecnorm = make_env_policy(cfg)

    frames_per_batch = env.num_envs * cfg.algo.train_every
    total_iters = cfg.max_iters if cfg.get("max_iters", -1) > 0 else cfg.total_frames // frames_per_batch
    save_interval = cfg.get("save_interval", -1)
    log_interval = (env.max_episode_length // cfg.algo.train_every) + 1

    stats_keys = [k for k in env.reward_spec.keys(True, True) if isinstance(k, tuple) and k[0] == "stats"]
    episode_stats = EpisodeStats(stats_keys, device=env.device)

    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    def save(checkpoint_name: str):
        ckpt_path = os.path.join(ckpt_dir, f"{checkpoint_name}.pt")
        state_dict = OrderedDict()
        state_dict["wandb"] = {"name": run.name, "id": run.id}
        state_dict["git_commit"] = git_commit
        state_dict["policy"] = policy.state_dict()
        state_dict["env"] = env.state_dict()
        state_dict["cfg"] = OmegaConf.to_container(cfg)
        state_dict["vecnorm"] = vecnorm.state_dict()
        torch.save(state_dict, ckpt_path)
        if cfg.wandb.mode != "disabled" and cfg.wandb.get("upload_checkpoints", True):
            run.save(ckpt_path, base_path=output_dir, policy="now")
        logging.info(f"Saved checkpoint to {ckpt_path}")

    assert env.training
    carry = env.reset()
    rollout_policy: TensorDictModuleBase = policy.get_rollout_policy("train")

    # one probe step to allocate the rollout buffer
    keep_next = ("done", "terminated", "discount", "reward", "stats", "is_init")
    with torch.inference_mode():
        tmp_carry = rollout_policy(carry.clone(False))
        tmp_td, _ = env.step_and_maybe_reset(tmp_carry.clone(False))
        tmp_td["next"] = tmp_td["next"].select(*keep_next, strict=False)

    N, T, device = env.num_envs, cfg.algo.train_every, env.device
    data_buf = TensorDict({}, batch_size=[N, T], device=device)
    for key, value in tmp_td.items(include_nested=True, leaves_only=True):
        data_buf.set(key, torch.empty((N, T, *value.shape[1:]), dtype=value.dtype, device=device))
    logging.info(f"Rollout buffer size: {data_buf.bytes() / 1e6:.2f} MB")

    env_frames = 0
    start_iter = env.current_iter  # > 0 when resuming from a checkpoint
    for i in tqdm(range(total_iters)):
        rollout_start = time.perf_counter()
        env.set_progress(start_iter + i)
        if hasattr(policy, "begin_iteration"):
            policy.begin_iteration()
        # the student rolls out its deterministic action during the pure-DAgger phase
        exploration = policy.exploration_type() if hasattr(policy, "exploration_type") else ExplorationType.RANDOM
        with torch.inference_mode(), set_exploration_type(exploration):
            for step in range(T):
                carry = rollout_policy(carry)
                td, carry = env.step_and_maybe_reset(carry)
                td["next"] = td["next"].select(*keep_next, strict=False)
                data_buf[:, step] = td
            policy.critic(data_buf)
            values = data_buf["state_value"]
            data_buf["next", "state_value"] = torch.where(
                data_buf["next", "done"],
                values,  # a workaround to avoid storing the next states
                torch.cat([values[:, 1:], policy.critic(carry.copy())["state_value"].unsqueeze(1)], dim=1),
            )
        rollout_time = time.perf_counter() - rollout_start

        episode_stats.add(data_buf)
        env_frames += data_buf.numel()

        info = {}
        if i % log_interval == 0 and len(episode_stats):
            for k, v in sorted(episode_stats.pop().items(True, True)):
                info["train/" + ("/".join(k) if isinstance(k, tuple) else k)] = torch.mean(v.float()).item()
        training_start = time.perf_counter()
        info.update(policy.train_op(data_buf))
        info.update(env.extra)
        info.update(env.stats_ema)
        info["env_frames"] = env_frames
        info["rollout_fps"] = data_buf.numel() / rollout_time
        info["training_time"] = time.perf_counter() - training_start
        run.log(info)
        if i % cfg.get("print_interval", log_interval) == 0:
            summary = {k: info[k] for k in PRINT_KEYS if k in info}
            print(f"[iter {start_iter + i}] " + " | ".join(f"{k}: {v:.4g}" for k, v in summary.items()), flush=True)

        if i > 0 and save_interval > 0 and i % save_interval == 0:
            save(f"checkpoint_{start_iter + i}")

    save("checkpoint_final")

    info = evaluate(env, policy.get_rollout_policy("eval"), render=cfg.eval_render, seed=cfg.seed)
    run.log(info)
    wandb.finish()
    # closing the Isaac Sim app can hang; exit directly once everything is saved and logged
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
