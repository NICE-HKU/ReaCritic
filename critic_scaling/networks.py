"""
The networks for ReLara algorithm.
"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

class BasicActorSAC(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.fc1 = nn.Linear(np.array(observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(action_space.shape))
        # action rescaling
        self.register_buffer("action_scale",
                             torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias",
                             torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32))
        self.log_std_max = 2
        self.log_std_min = -5

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        return mean, log_std

    def get_action(self, x):
        x = x.to(torch.float32)
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.ReLU()):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.activation = activation

    def forward(self, x):
        residual = x
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        x += residual
        x = self.activation(x)
        return x


class QNetworkResidual(nn.Module):
    def __init__(self, observation_space, action_space, block_num=3):
        super().__init__()
        self.fc1 = nn.Linear(np.array(observation_space.shape).prod() + np.prod(action_space.shape), 256)
        self.hidden_blocks = nn.ModuleList([ResidualBlock(256, 256) for _ in range(block_num)])
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.fc1(x)
        for block in self.hidden_blocks:
            x = block(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ActorResidual(nn.Module):

    def __init__(self, observation_space, action_space, MLP, block_num=3):
        super().__init__()
        self.fc1 = nn.Linear(np.array(observation_space.shape).prod(), 256)
        self.hidden_blocks = nn.ModuleList([ResidualBlock(256, 256) for _ in range(block_num)])
        self.fc2 = nn.Linear(256, 128)
        self.fc_mean = nn.Linear(128, np.prod(action_space.shape))
        self.fc_logstd = nn.Linear(128, np.prod(action_space.shape))
        # action rescaling
        self.register_buffer("action_scale",
                             torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias",
                             torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32))
        self.log_std_max = 2
        self.log_std_min = -5

    def forward(self, x):
        x = self.fc1(x)
        for block in self.hidden_blocks:
            x = block(x)
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean



class BasicQNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.fc1 = nn.Linear(np.array(observation_space.shape).prod() + np.prod(action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReaCritic(nn.Module):
    def __init__(self, observation_space, action_space, HRstep, VRstep,
                 h_dim=256, n_heads=4, drop_p=0.1, noise_std=0.01):
        super().__init__()
        self.state_dim = np.prod(observation_space.shape)
        self.act_dim = np.prod(action_space.shape)
        self.h_dim = h_dim
        self.reasoning_steps = HRstep
        self.noise_std = noise_std

        
        self.embed_sa = nn.Linear(self.state_dim + self.act_dim, h_dim)

      
        self.step_embed = nn.Embedding(HRstep, h_dim)

        self.blocks = nn.ModuleList([
            ReaBlock(h_dim, n_heads, drop_p)
            for _ in range(VRstep)
        ])

        # Attention over steps
        self.attn_over_steps = nn.Linear(h_dim, 1)


        self.predict_q = nn.Sequential(
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, 1)
        )

    def forward(self, x, a):
        x = x.to(torch.float32)
        a = a.to(torch.float32)
        B = x.shape[0]


        sa_combined = torch.cat([x, a], dim=1)
        sa_embed = self.embed_sa(sa_combined)  # [B, h_dim]


        reasoning_seq = sa_embed.unsqueeze(1).expand(B, self.reasoning_steps, self.h_dim)  # [B, steps, h_dim]
        step_ids = torch.arange(self.reasoning_steps, device=x.device)
        step_emb = self.step_embed(step_ids).unsqueeze(0)  # [1, steps, h_dim]
        reasoning_seq = reasoning_seq + step_emb


        if self.noise_std > 0:
            noise = torch.randn_like(reasoning_seq) * self.noise_std
            reasoning_seq = reasoning_seq + noise

        # Transformer Blocks
        for block in self.blocks:
            reasoning_seq = block(reasoning_seq)

        # Attention over steps
        attn_weights = self.attn_over_steps(reasoning_seq)  # [B, steps, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        final_repr = torch.sum(reasoning_seq * attn_weights, dim=1)  # [B, h_dim]


        q_value = self.predict_q(final_repr)
        return q_value


class ReaBlock(nn.Module):
    def __init__(self, h_dim, n_heads, drop_p):
        super().__init__()
        self.ln1 = nn.LayerNorm(h_dim)
        self.attention = ReaAttention(h_dim, n_heads, drop_p)
        self.ln2 = nn.LayerNorm(h_dim)
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 4 * h_dim),
            nn.GELU(),
            nn.Linear(4 * h_dim, h_dim),
            nn.Dropout(drop_p),
        )

    def forward(self, x):
        # Pre-LN architecture for better training stability
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class ReaAttention(nn.Module):
    def __init__(self, h_dim, n_heads, drop_p):
        super().__init__()
        assert h_dim % n_heads == 0

        self.h_dim = h_dim
        self.n_heads = n_heads
        self.head_dim = h_dim // n_heads

        self.qkv = nn.Linear(h_dim, 3 * h_dim)
        self.proj = nn.Linear(h_dim, h_dim)
        self.drop = nn.Dropout(drop_p)

    def forward(self, x):
        B, T, C = x.shape

        # 
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, T, D]

        # Standard scaled dot-product attention
        # 
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale  # [B, H, T, T]

        # 
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)

        return x