"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        # Keep the original channel counts
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Global average pooling and final dense layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)
        
    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        # Second block
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # Global average pooling
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Final classification
        logits = self.fc(x)
        probabilities = F.softmax(logits, dim=-1)
        
        return logits, probabilities
class SmallerComparableCNN(nn.Module):
    def __init__(self):
        super(SmallerComparableCNN, self).__init__()
        # Reduced initial channels and total layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Reduced from 64 to 32
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Reduced from 128 to 64
        self.bn2 = nn.BatchNorm2d(64)
        
        # Global average pooling and final dense layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)  # Changed input features to match last conv layer
        
    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        # Second block
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # Global average pooling
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Final classification
        logits = self.fc(x)
        probabilities = F.softmax(logits, dim=-1)
        
        return logits, probabilities

@dataclass
class ModelArgss:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: int = 0
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)

@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: int = 0
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)
        
        # Add this for image processing
        self.seq_len = 256  # For CIFAR-10: (32//2 * 32//2) after initial conv

class Mamba(nn.Module):
    def __init__(self, args: ModelArgs, num_classes: int):
        """Full Mamba model."""
        super().__init__()
        self.args = args
        
        # Embedding layer
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        
        # Residual blocks
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        
        # Normalization layer
        self.norm_f = RMSNorm(args.d_model)

        # Fully connected layer for classification
        self.fc = nn.Linear(args.d_model, num_classes)

    def forward(self, input_ids):
        """
        Args:
            input_ids (long tensor): shape (b, l)

        Returns:
            logits: shape (b, num_classes)
            probabilities: shape (b, num_classes) (Softmax probabilities)
        """
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)

        # Take the last time step output for classification
        x = x[:, -1, :]

        logits = self.fc(x)
        probabilities = F.softmax(logits, dim=-1)  # Compute softmax probabilities

        return logits, probabilities  # Return both logits and probabilities

    
    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.
    
        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'
                            
        Returns:
            model: Mamba model with weights loaded
    
        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
        
        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))
        
        
        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )
        model = Mamba(args)
        
        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)
        
        return model
    
class ProperImageMamba(nn.Module):
    def __init__(self, args: ModelArgs, num_classes: int):
        super().__init__()
        self.args = args
        
        # Initial conv layers to process image
        self.conv1 = nn.Conv2d(3, args.d_model, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(args.d_model)
        
        # Convert 2D feature maps to sequence
        self.patch_to_seq = nn.Sequential(
            nn.Conv2d(args.d_model, args.d_model, kernel_size=2, stride=2),
            nn.BatchNorm2d(args.d_model),
            nn.GELU()
        )
        
        # Calculate sequence length from image size
        # For CIFAR-10 (32x32), after one 2x2 stride conv: 16x16 = 256 sequence length
        self.seq_len = (32 // 2) * (32 // 2)  # 256 for CIFAR-10
        
        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(args) for _ in range(args.n_layer)
        ])
        
        # Output layers
        self.norm = RMSNorm(args.d_model)
        self.fc = nn.Linear(args.d_model, num_classes)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)  # [B, d_model, 32, 32]
        x = self.bn1(x)
        x = F.gelu(x)
        
        # Convert to sequence
        x = self.patch_to_seq(x)  # [B, d_model, 16, 16]
        
        # Reshape to sequence
        B, C, H, W = x.shape
        x = x.view(B, C, -1)  # [B, d_model, seq_len]
        x = x.permute(0, 2, 1)  # [B, seq_len, d_model]
        
        # Apply Mamba blocks
        for layer in self.layers:
            x = layer(x)
        
        # Global pooling (mean across sequence length)
        x = x.mean(dim=1)  # [B, d_model]
        
        # Final classification
        x = self.norm(x)
        logits = self.fc(x)
        probabilities = F.softmax(logits, dim=-1)
        
        return logits, probabilities

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # Input projection
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        
        # Convolution for local mixing
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            kernel_size=args.d_conv,
            padding=args.d_conv - 1,
            groups=args.d_inner,
            bias=args.conv_bias
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        
        # Initialize A and D for stable selective SSM
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        # Split into x and residual
        x_and_res = self.in_proj(x)
        x, res = x_and_res.chunk(2, dim=-1)
        
        # Convolutional mixing
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :self.args.seq_len]
        x = rearrange(x, 'b d l -> b l d')
        
        # Apply selective SSM
        x = self.ssm(x)
        
        # Combine with residual and project
        x = x * F.silu(res)
        output = self.out_proj(x)
        
        return output

    def ssm(self, x):
        """Selective State Space computation"""
        d_inner = self.args.d_inner
        
        # Get A, D parameters
        A = -torch.exp(self.A_log)
        D = self.D
        
        # Project input to get Δ, B, C
        x_dbl = self.x_proj(x)
        delta, B, C = x_dbl.split(
            [self.args.dt_rank, self.args.d_state, self.args.d_state], 
            dim=-1
        )
        
        # Process delta through projection
        delta = F.softplus(self.dt_proj(delta))
        
        # Perform selective scan
        return self.selective_scan(x, delta, A, B, C, D)
        
    def selective_scan(self, u, delta, A, B, C, D):
        """Parallel scan implementation of SSM"""
        # Discretize continuous parameters
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        # Initialize state
        x = torch.zeros(
            (u.shape[0], self.args.d_inner, self.args.d_state),
            device=deltaA.device
        )
        
        # Parallel scan
        ys = []
        for i in range(u.shape[1]):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
            
        y = torch.stack(ys, dim=1)
        
        return y + u * D

class ImageMamba(nn.Module):
    def __init__(self, args: ModelArgs, num_classes: int):
        super().__init__()
        self.args = args
        # Define conv layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=args.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=args.d_model, out_channels=args.d_model * 2, kernel_size=3, padding=1)
        # Residual blocks
        self.layers = nn.ModuleList([ImageResidualBlock(args) for _ in range(args.n_layer)])
        # Normalization layer
        self.norm_f = nn.BatchNorm2d(args.d_model * 2)
        # Pooling and FC layers
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(args.d_model * 2, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probabilities = F.softmax(logits, dim=-1)
        return logits, probabilities
    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.
    
        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'
                            
        Returns:
            model: Mamba model with weights loaded
    
        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
        
        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))
        
        
        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )
        model = Mamba(args)
        
        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)
        
        return model

class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)
        

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        output = self.mixer(self.norm(x)) + x

        return output
            
class ImageResidualBlock(nn.Module):
    def __init__(self, args):
        super(ImageResidualBlock, self).__init__()
        self.eps = 1e-6
        self.weight = nn.Parameter(torch.ones(args.d_model * 2))  # Properly initialize self.weight
        self.conv = nn.Conv2d(args.d_model * 2, args.d_model * 2, kernel_size=3, padding=1)

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.view(1, -1, 1, 1)
        return output

class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
        