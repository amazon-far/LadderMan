"""Visualize / evaluate a trained policy and optionally export it to ONNX for deployment.

Examples:
    python scripts/play.py algo=student task=ladder/student checkpoint_path=<path/to/checkpoint.pt>
    python scripts/play.py algo=student task=ladder/student checkpoint_path=<...> export_policy=true headless=true
"""
import copy
import itertools
import os
import re
import sys
import time

import hydra
import torch
import yaml
from omegaconf import OmegaConf
from tensordict.nn import TensorDictSequential
from torchrl.envs.utils import ExplorationType, set_exploration_type

from isaaclab.app import AppLauncher

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(FILE_PATH, ".."))
sys.path.insert(0, REPO_ROOT)

import active_adaptation  # noqa: E402,F401  (registers the algorithm configs with Hydra)


def export(cfg, env, policy, vecnorm):
    from active_adaptation.assets import get_asset_meta
    from active_adaptation.utils.checkpoint import load_checkpoint
    from active_adaptation.utils.export import export_onnx
    from active_adaptation.utils.wandb import parse_checkpoint_path
    from scripts.helpers import ObsNorm, ObsOODDetector

    checkpoint_path = parse_checkpoint_path(cfg.checkpoint_path, project=cfg.wandb.project)
    run_id, checkpoint_num = "unknown", "unknown"
    if checkpoint_path is not None:
        state_dict = load_checkpoint(checkpoint_path)
        run_id = (state_dict.get("wandb") or {}).get("id") or "unknown"
        match = re.search(r"checkpoint_(\d+|final)", os.path.basename(checkpoint_path))
        if match:
            checkpoint_num = match.group(1)

    fake_input = env.observation_spec[0].rand().cpu()
    fake_input["is_init"] = torch.tensor(1, dtype=bool)
    fake_input = fake_input.unsqueeze(0)

    deploy_policy = copy.deepcopy(policy.get_rollout_policy("deploy"))
    obs_norm = ObsNorm.from_vecnorm(vecnorm, deploy_policy.in_keys)
    ood_detector = ObsOODDetector(deploy_policy.in_keys, sigma=5.0)
    _policy = TensorDictSequential(obs_norm, ood_detector, deploy_policy).cpu()

    start = time.perf_counter()
    for _ in range(100):
        _policy(fake_input)
    print(f"Inference time of policy: {(time.perf_counter() - start) / 100 * 1e3:.3f} ms")

    export_dir = os.path.join(REPO_ROOT, "exports", cfg.task.name)
    os.makedirs(export_dir, exist_ok=True)
    path = os.path.join(export_dir, f"policy-{run_id}-{checkpoint_num}.onnx")
    export_onnx(_policy, fake_input, path, meta={})

    # deployment config: observations, action scaling, PD gains, joint ordering, reference motion
    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    policy_config = {"observation": {k: dict_cfg["task"]["observation"][k] for k in deploy_policy.in_keys}}
    policy_config["action_scale"] = dict_cfg["task"]["action"]["action_scaling"]

    asset_meta = get_asset_meta(env.scene["robot"])
    policy_config["isaac_joint_names"] = asset_meta["joint_names_isaac"]
    joint_kp, joint_kd = {}, {}
    for actuator in asset_meta["actuators"].values():
        for gains, value in ((joint_kp, actuator["stiffness"]), (joint_kd, actuator["damping"])):
            if isinstance(value, float):
                exprs = actuator["joint_names_expr"]
                for joint_name in (exprs if isinstance(exprs, list) else [exprs]):
                    gains[joint_name] = value
            elif isinstance(value, dict):
                gains.update(value)
            else:
                raise ValueError(f"Unsupported gain type: {type(value)}")
    policy_config["joint_kp"] = joint_kp
    policy_config["joint_kd"] = joint_kd
    policy_config["default_joint_pos"] = asset_meta["init_state"]["joint_pos"]
    policy_config["policy_joint_names"] = env.action_manager.joint_names

    command = env.command_manager
    assert command.dataset.num_motions == 1
    for obs_cfg in policy_config["observation"].get("command", {}).values():
        obs_cfg["motion_duration_second"] = command.dataset.lengths[0].item() * env.step_dt
        obs_cfg["motion_path"] = cfg.task.command.data_path
        obs_cfg["future_steps"] = command.future_steps.tolist()
        obs_cfg["body_names"] = command.tracking_keypoint_names
        obs_cfg["joint_names"] = command.tracking_joint_names
        obs_cfg["root_body_name"] = command.root_body_name

    with open(path.replace(".onnx", ".yaml"), "w") as f:
        yaml.dump(policy_config, f, sort_keys=False)
    print(f"Policy config saved to {path.replace('.onnx', '.yaml')}")


@hydra.main(config_path=os.path.join(REPO_ROOT, "cfg"), config_name="play", version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app))
    simulation_app = app_launcher.app

    from scripts.helpers import EpisodeStats, make_env_policy
    env, policy, vecnorm = make_env_policy(cfg)

    if cfg.export_policy:
        export(cfg, env, policy, vecnorm)

    stats_keys = [k for k in env.reward_spec.keys(True, True) if isinstance(k, tuple) and k[0] == "stats"]
    episode_stats = EpisodeStats(stats_keys, device=env.device)
    rollout_policy = policy.get_rollout_policy("eval")

    env.base_env.eval()
    td_ = env.reset()
    max_steps = cfg.get("max_steps", -1)
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for i in itertools.count():
            if 0 < max_steps <= i:
                break
            td_ = rollout_policy(td_)
            td, td_ = env.step_and_maybe_reset(td_)
            episode_stats.add(td)
            if len(episode_stats) >= env.num_envs:
                print("Step", i)
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    print(k, torch.mean(v).item())

    # closing the Isaac Sim app can hang; exit directly
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
