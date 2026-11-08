# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



from typing import Sequence

import torch
import torch.nn as nn
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase as ModBase
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
from torchrl.modules import ProbabilisticActor

from ..modules.distributions import IndependentNormal

# observation groups (see the task YAML)
CMD_KEY = "command"          # reference-motion command
OBS_KEY = "policy"           # proprioception
OBS_PRIV_KEY = "priv"        # privileged state (teacher / critic only)
OBJECT_KEY = "object"        # ladder pose in the robot frame (teacher / critic only)
DEPTH_KEY = "depth"          # depth image (student only)
REF_JPOS_KEY = "ref_joint_pos_"  # reference joint positions for the teacher's residual action

ACTION_KEY = "action"
REWARD_KEY = ("next", "reward")
TERM_KEY = ("next", "terminated")
DONE_KEY = ("next", "done")


def make_mlp(num_units, activation=nn.Mish, norm="before", dropout=0.):
    assert norm in ("before", "after", None)
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        if norm == "before":
            layers.append(nn.LayerNorm(n))
            layers.append(activation())
        elif norm == "after":
            layers.append(activation())
            layers.append(nn.LayerNorm(n))
        else:
            layers.append(activation())
        if dropout > 0. :
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def make_conv(num_channels, activation=nn.LeakyReLU, kernel_sizes=3, flatten: bool=True):
    layers = []
    if isinstance(kernel_sizes, int):
        kernel_sizes = [kernel_sizes] * len(num_channels)
    for n, k in zip(num_channels, kernel_sizes):
        layers.append(nn.LazyConv2d(n, kernel_size=k, stride=2, padding=k//2))
        layers.append(activation())
    if flatten:
        layers.append(nn.Flatten())
    return FlattenBatch(nn.Sequential(*layers), data_dim=3)


class FlattenBatch(nn.Module):
    def __init__(self, module, data_dim: int=1):
        super().__init__()
        self.module = module
        self.data_dim = data_dim

    def forward(self, input: torch.Tensor):
        batch_shape = input.shape[:-self.data_dim]
        flattened_input = input.flatten(0, len(batch_shape)-1)
        output = self.module(flattened_input)
        return output.unflatten(0, batch_shape)


def make_batch(tensordict: TensorDict, num_minibatches: int, seq_len: int = -1):
    if seq_len > 1:
        N, T = tensordict.shape
        T = (T // seq_len) * seq_len
        tensordict = tensordict[:, :T].reshape(-1, seq_len)
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices]
    else:
        tensordict = tensordict.reshape(-1)
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices]


class Actor(nn.Module):
    """Linear Gaussian head with a state-independent std.

    When ``load_noise_scale`` is set, the std is reset to that value whenever a
    state dict is loaded (e.g. when finetuning from a checkpoint).
    """
    def __init__(self, action_dim: int, init_noise_scale: float=1.0, predict_std: bool=False, load_noise_scale: float | None=None) -> None:
        super().__init__()
        self.predict_std = predict_std
        if predict_std:
            self.actor_mean = nn.LazyLinear(action_dim * 2)
        else:
            self.actor_mean = nn.LazyLinear(action_dim)
            self.actor_std = nn.Parameter(torch.ones(action_dim) * init_noise_scale)
        self.scale_mapping = nn.Identity()
        self.load_noise_scale = load_noise_scale
    
    def forward(self, features: torch.Tensor):
        if self.predict_std:
            loc, scale = self.actor_mean(features).chunk(2, dim=-1)
        else:
            loc = self.actor_mean(features)
            scale = torch.ones_like(loc) * self.actor_std
        scale = self.scale_mapping(scale)
        return loc, scale
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        if self.load_noise_scale is not None:
            self.actor_std.data.fill_(self.load_noise_scale)


class GAE(nn.Module):
    def __init__(self, gamma, lmbda):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.gamma: torch.Tensor
        self.lmbda: torch.Tensor
    
    def forward(
        self, 
        reward: torch.Tensor, 
        terminated: torch.Tensor,
        done: torch.Tensor, 
        value: torch.Tensor, 
        next_value: torch.Tensor,
        discount: torch.Tensor=None
    ):
        num_steps = terminated.shape[1]
        advantages = torch.zeros_like(reward)
        nonterm = 1 - terminated.float() # whether to backup value
        nondone = 1 - done.float()       # whether to backup reward
        if discount is None:
            discount = torch.ones_like(nonterm)
        gae = 0
        for step in reversed(range(num_steps)):
            next_value_t = next_value[:, step] * nonterm[:, step]
            gamma_t = discount[:, step] * self.gamma
            delta = reward[:, step] + gamma_t * next_value_t - value[:, step]
            advantages[:, step] = gae = delta + (gamma_t * self.lmbda * nondone[:, step] * gae)
        returns = advantages + value
        return advantages, returns


class CatTensors(ModBase):
    def __init__(self, in_keys, out_key, del_keys=False, sort=True):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = [out_key]

        self.del_keys = del_keys
        self.sort = sort
        if self.sort:
            self.in_keys = sorted(self.in_keys)

    def forward(self, tensordict: TensorDictBase):
        out = torch.cat([tensordict.get(k) for k in self.in_keys], dim=-1)
        tensordict.set(self.out_keys[0], out)
        if self.del_keys:
            tensordict.exclude(*self.in_keys, inplace=True)
        return tensordict


# ---------------------------------------------------------------------------
# Network builders shared by the teacher and the student.
# The module layout (and hence the state-dict keys) is kept identical to the
# networks used to train the released checkpoints.
# ---------------------------------------------------------------------------

class RefJointPos(nn.Module):
    """Residual action: the policy outputs an offset on top of the reference joint positions."""
    def forward(self, ref_jpos, action):
        return (ref_jpos + action,)


def build_actor(
    in_keys: Sequence[str],
    action_dim: int,
    init_noise_scale: float,
    load_noise_scale: float | None,
    device,
    residual: bool = False,
) -> ProbabilisticActor:
    dist_keys = IndependentNormal.dist_keys
    actor_modules = [
        CatTensors(list(in_keys), "_actor_inp", del_keys=False, sort=False),
        Mod(make_mlp([512, 256, 256]), ["_actor_inp"], ["_actor_feature"]),
        Mod(Actor(action_dim, init_noise_scale=init_noise_scale, load_noise_scale=load_noise_scale), ["_actor_feature"], dist_keys),
    ]
    if residual:
        actor_modules.append(Mod(RefJointPos(), [REF_JPOS_KEY, "loc"], ["loc"]))
    return ProbabilisticActor(
        module=Seq(*actor_modules),
        in_keys=dist_keys,
        out_keys=[ACTION_KEY],
        distribution_class=IndependentNormal,
        return_log_prob=True,
    ).to(device)


def build_priv_encoder(in_keys: Sequence[str], latent_dim: int, out_key: str, device) -> Seq:
    return Seq(
        CatTensors(list(in_keys), "_encoder_priv_inp", del_keys=False, sort=False),
        Mod(nn.Sequential(make_mlp([latent_dim]), nn.LazyLinear(latent_dim)), "_encoder_priv_inp", out_key),
        selected_out_keys=[out_key],
    ).to(device)


def build_critic(in_keys: Sequence[str], num_reward_groups: int, device) -> Seq:
    _critic = nn.Sequential(make_mlp([512, 256, 128]), nn.LazyLinear(num_reward_groups))
    return Seq(
        CatTensors(list(in_keys), "_critic_input", del_keys=False),
        Mod(_critic, ["_critic_input"], ["state_value"]),
    ).to(device)


def build_depth_encoder(out_key: str, device) -> Mod:
    depth_cnn = nn.Sequential(
        make_conv(num_channels=[8, 8, 8], activation=nn.Mish, kernel_sizes=5),
        nn.LazyLinear(64),
        nn.LayerNorm(64),
    )
    return Mod(depth_cnn, [DEPTH_KEY], [out_key]).to(device)


def orthogonal_init_(module: nn.Module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, 0.01)
        nn.init.constant_(module.bias, 0.)


def actor_std(actor: ProbabilisticActor) -> torch.Tensor:
    return actor.module[0][2].module.actor_std
