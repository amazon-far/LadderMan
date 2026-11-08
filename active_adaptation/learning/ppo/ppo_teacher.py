"""Step 1: privileged teacher, trained with PPO (one teacher per ladder inclination).

The teacher observes the reference-motion command, proprioception, privileged state
and the ladder pose, and outputs a *residual* joint-position action on top of the
reference joint positions (``ref_joint_pos_``).

Networks (state-dict keys are compatible with the released teacher checkpoints):
    encoder_priv: [priv, object]                  -> priv_feature
    actor:        [command, policy, priv_feature] -> residual action
    critic:       [command, object, policy, priv] -> value per reward group
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.utils._pytree as pytree
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential as Seq
from torchrl.data import CompositeSpec, TensorSpec

from ..modules.distributions import IndependentNormal
from ..utils.valuenorm import ValueNorm1, ValueNormFake
from .common import (
    ACTION_KEY, CMD_KEY, DONE_KEY, OBJECT_KEY, OBS_KEY, OBS_PRIV_KEY, REF_JPOS_KEY, REWARD_KEY, TERM_KEY,
    GAE, actor_std, build_actor, build_critic, build_priv_encoder, make_batch, orthogonal_init_,
)

torch.set_float32_matmul_precision("high")

PRIV_FEATURE_KEY = "priv_feature"


@dataclass
class TeacherConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_teacher.PPOTeacher"
    name: str = "ppo_teacher"
    max_iters: int = -1  # number of training iterations (<= 0: use `total_frames`)

    train_every: int = 32
    ppo_epochs: int = 3
    num_minibatches: int = 8
    clip_param: float = 0.2
    gamma: float = 0.99
    lmbda: float = 0.95
    lr: float = 3e-4
    desired_kl: float | None = 0.01  # adaptive learning rate; None for a constant lr
    entropy_coef_start: float = 0.001
    entropy_coef_end: float = 0.001
    entropy_decay_iters: int = 1000
    init_noise_scale: float = 1.0
    load_noise_scale: float | None = 0.5
    max_grad_norm: float = 1.0
    latent_dim: int = 256
    residual_action: bool = True
    value_norm: bool = False
    normalize_before_sum: bool = False
    clip_neg_reward: bool = False
    normalize_ratio: bool = False

    # observation normalizer mode: "train" (update running stats) or "eval" (frozen)
    vecnorm: str | None = "eval"
    # observation groups kept in the env (groups ending with "_" are always kept)
    in_keys: List[str] = field(default_factory=lambda: [CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY])


cs = ConfigStore.instance()
cs.store("ppo_teacher", node=TeacherConfig(), group="algo")


def build_teacher_networks(cfg, observation_spec: CompositeSpec, action_dim: int, device):
    """Return ``(encoder_priv, actor)`` with the teacher architecture."""
    encoder_in_keys = [OBS_PRIV_KEY]
    if observation_spec.get(OBJECT_KEY, None) is not None:
        encoder_in_keys.append(OBJECT_KEY)
    if cfg.residual_action:
        assert REF_JPOS_KEY in observation_spec.keys(), f"{REF_JPOS_KEY} observation is required for the residual action"
    encoder_priv = build_priv_encoder(encoder_in_keys, cfg.latent_dim, PRIV_FEATURE_KEY, device)
    actor = build_actor(
        [CMD_KEY, OBS_KEY, PRIV_FEATURE_KEY], action_dim,
        init_noise_scale=cfg.init_noise_scale, load_noise_scale=cfg.load_noise_scale,
        device=device, residual=cfg.residual_action,
    )
    return encoder_priv, actor


class PPOTeacher(TensorDictModuleBase):
    def __init__(
        self,
        cfg: TeacherConfig,
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

        self.entropy_coef = cfg.entropy_coef_start
        self.desired_kl = cfg.desired_kl
        self.clip_param = cfg.clip_param
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.gae = GAE(gamma=cfg.gamma, lmbda=cfg.lmbda)
        self.reward_groups = list(env.cfg.reward.keys())
        num_reward_groups = len(self.reward_groups)
        self.reward_scales = torch.ones(num_reward_groups, device=self.device) / num_reward_groups
        self.value_norm = (ValueNorm1 if cfg.value_norm else ValueNormFake)(input_shape=num_reward_groups).to(self.device)

        self.action_dim = action_spec.shape[-1]
        self.joint_names = env.action_manager.joint_names
        self.dist_cls = IndependentNormal
        self.dist_keys = IndependentNormal.dist_keys

        self.encoder_priv, self.actor = build_teacher_networks(cfg, observation_spec, self.action_dim, device)
        critic_in_keys = [OBS_PRIV_KEY, OBS_KEY, CMD_KEY]
        if observation_spec.get(OBJECT_KEY, None) is not None:
            critic_in_keys.append(OBJECT_KEY)
        self.critic = build_critic(critic_in_keys, num_reward_groups, device)

        fake_input = observation_spec.zero()
        self.encoder_priv(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)
        self.apply(orthogonal_init_)

        self.lr_policy = cfg.lr
        self.opt_policy = torch.optim.Adam(
            [{"params": self.actor.parameters()}, {"params": self.encoder_priv.parameters()}],
            lr=self.lr_policy,
        )
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.num_updates = 0

    def get_rollout_policy(self, mode: str = "train"):
        out_keys = ["sample_log_prob", "action"] + self.dist_keys
        return Seq(self.encoder_priv, self.actor, selected_out_keys=out_keys)

    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.exclude("stats")
        info = self.train_policy(tensordict.copy())
        self.num_updates += 1
        std = actor_std(self.actor).detach()
        for joint_name, s in zip(self.joint_names, std):
            info[f"actor_std/{joint_name}"] = s
        info["actor_std/mean"] = std.mean()
        return info

    def train_policy(self, tensordict: TensorDict):
        infos = []
        self._compute_advantage(tensordict, self.critic, "adv", "ret", update_value_norm=True)

        entropy_progress = float(np.clip(self.env.current_iter / self.cfg.entropy_decay_iters, 0., 1.))
        self.entropy_coef = self.cfg.entropy_coef_start + (self.cfg.entropy_coef_end - self.cfg.entropy_coef_start) * entropy_progress

        for epoch in range(self.cfg.ppo_epochs):
            for minibatch in make_batch(tensordict, self.cfg.num_minibatches):
                infos.append(self._update_ppo(minibatch))
                if self.desired_kl is not None:  # adaptive learning rate
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
            infos[f"critic/{group_name}.neg_rew_ratio"] = (tensordict[REWARD_KEY][:, :, i] <= 0.).float().mean().item()
        return dict(sorted(infos.items()))

    @torch.no_grad()
    def _compute_advantage(self, tensordict: TensorDict, critic, adv_key="adv", ret_key="ret", update_value_norm=True):
        keys = tensordict.keys(True, True)
        if not ("state_value" in keys and ("next", "state_value") in keys):
            with tensordict.view(-1) as tensordict_flat:
                critic(tensordict_flat)
                critic(tensordict_flat["next"])

        values = self.value_norm.denormalize(tensordict["state_value"])
        next_values = self.value_norm.denormalize(tensordict["next", "state_value"])
        rewards = tensordict[REWARD_KEY]
        if self.cfg.clip_neg_reward:
            rewards = rewards.clamp_min(0.)
        adv, ret = self.gae(rewards, tensordict[TERM_KEY], tensordict[DONE_KEY], values, next_values, tensordict["next", "discount"])

        if self.cfg.normalize_before_sum:  # normalize, scale, sum
            adv_norm = (adv - adv.mean(dim=(0, 1))) / (adv.std(dim=(0, 1)) + 0.01)
            adv_final = (adv_norm * self.reward_scales).sum(dim=2, keepdim=True)
        else:  # scale, sum, normalize
            adv_sum = (adv * self.reward_scales).sum(dim=2, keepdim=True)
            adv_final = (adv_sum - adv_sum.mean(dim=(0, 1))) / (adv_sum.std(dim=(0, 1)) + 1e-8)

        if update_value_norm:
            self.value_norm.update(ret)
        tensordict.set(adv_key, adv_final)
        tensordict.set(ret_key, self.value_norm.normalize(ret))
        return tensordict

    def _update_ppo(self, tensordict: TensorDict):
        dist_kwargs_old = tensordict.select(*self.dist_keys)
        self.encoder_priv(tensordict)
        dist: D.Independent = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy().mean()
        valid = (tensordict["step_count"] > 1).squeeze(-1)

        adv = tensordict["adv"]
        log_ratio = (log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        ratio = torch.exp(log_ratio)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1. - self.clip_param, 1. + self.clip_param)
        if self.cfg.normalize_ratio:
            clamped_ratio = ratio.clamp(1. - self.clip_param, 1. + self.clip_param).detach()
            surr1, surr2 = surr1 / clamped_ratio, surr2 / clamped_ratio
        policy_loss = -(torch.min(surr1, surr2)[valid]).mean()
        entropy_loss = -self.entropy_coef * entropy

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)[valid].mean(dim=0)

        loss = policy_loss + entropy_loss + value_loss.mean()
        self.opt_policy.zero_grad()
        self.opt_critic.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        priv_grad_norm = nn.utils.clip_grad_norm_(self.encoder_priv.parameters(), self.cfg.max_grad_norm)
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
            "actor/clamp_ratio": clipfrac,
            "actor/kl": kl,
            "actor/priv_grad_norm": priv_grad_norm,
            "actor/approx_kl": ((ratio - 1) - log_ratio).mean(),
            "critic/grad_norm": critic_grad_norm,
        }
        for i, group_name in enumerate(self.reward_groups):
            info[f"critic/{group_name}.explained_var"] = explained_var[i]
            info[f"critic/{group_name}.value_loss"] = value_loss[i].detach()
        return info

    def state_dict(self):
        state_dict = OrderedDict()
        for name in ("encoder_priv", "actor", "critic", "value_norm"):
            state_dict[name] = getattr(self, name).state_dict()
        state_dict["last_iter"] = self.env.current_iter
        state_dict["lr_policy"] = self.lr_policy
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        failed = []
        for name in ("encoder_priv", "actor", "critic", "value_norm"):
            if name not in state_dict:
                failed.append(name)
                continue
            try:
                getattr(self, name).load_state_dict(state_dict[name], strict=strict)
            except Exception as e:
                warnings.warn(f"Failed to load state dict for {name}: {e}")
                failed.append(name)
        if failed:
            warnings.warn(f"Teacher modules not loaded from checkpoint: {failed}")
        self.env.set_progress(state_dict.get("last_iter", 0))
        lr_policy = state_dict.get("lr_policy", None)
        if lr_policy is not None:
            self.lr_policy = lr_policy
            for param_group in self.opt_policy.param_groups:
                param_group["lr"] = self.lr_policy
        return failed
