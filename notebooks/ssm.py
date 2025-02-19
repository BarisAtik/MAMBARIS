import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import selective_scan_fn
from einops import rearrange, repeat
import math
from dataclasses import dataclass
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
    seq_len: int = 32*32  # Default sequence length
   
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
       
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
           
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)
       
        # Ensure seq_len is set to 256 for CIFAR-10
        self.seq_len = 256  # For CIFAR-10: (32//2 * 32//2) after initial conv
class FastMambaBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank=None,
        d_inner=None,
        seq_len=256
    ):
        super().__init__()
        
        # Model dimensions
        self.d_model = d_model
        self.d_inner = d_inner if d_inner is not None else int(expand * d_model)
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = math.ceil(self.d_inner / 16) if dt_rank is None else dt_rank
        self.seq_len = seq_len
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # Convolution for local mixing
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding='same',
            groups=self.d_inner
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        
        # Initialize A and D
        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A.float()))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x):
        # Ensure input type
        x = x.to(dtype=torch.float32)
        
        # Input projection and split
        x_and_res = self.in_proj(x)
        x, res = x_and_res.chunk(2, dim=-1)
        
        # Convolutional mixing
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)
        x = rearrange(x, 'b d l -> b l d')
        
        # SSM computation
        x_dbl = self.x_proj(x)
        d_dt, B_ssm, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # Process delta with explicit type
        delta = F.softplus(self.dt_proj(d_dt)).to(dtype=torch.float32)
        
        # Prepare A with explicit type
        A = -torch.exp(self.A_log).to(dtype=torch.float32)
        
        # Reshape for scan with explicit types
        u = rearrange(x, 'b l d -> b d l').to(dtype=torch.float32)
        delta = rearrange(delta, 'b l d -> b d l').to(dtype=torch.float32)
        B_ssm = rearrange(B_ssm, 'b l d -> b d l').to(dtype=torch.float32)
        C = rearrange(C, 'b l d -> b d l').to(dtype=torch.float32)
        D = self.D.to(dtype=torch.float32)
        
        # Selective scan
        x = selective_scan_fn(u, delta, A, B_ssm, C, D)
        
        # Back to original shape
        x = rearrange(x, 'b d l -> b l d')
        
        # Residual connection
        x = x * F.silu(res)
        
        return self.out_proj(x)

class FastImageMamba(nn.Module):
    def __init__(self, args: ModelArgs, num_classes: int):
        super().__init__()
        self.args = args
        
        # Initial image processing with smaller channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, args.d_model//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(args.d_model//2),
            nn.GELU(),
            nn.Conv2d(args.d_model//2, args.d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(args.d_model),
            nn.GELU()
        )
        
        # Patch to sequence with gradual downsampling
        self.patch_to_seq = nn.Sequential(
            nn.Conv2d(args.d_model, args.d_model, kernel_size=2, stride=2),
            nn.BatchNorm2d(args.d_model),
            nn.GELU(),
            nn.Dropout(0.1)  # Add dropout for regularization
        )
        
        # Mamba layers with residual connections
        self.layers = nn.ModuleList([
            nn.Sequential(
                FastMambaBlock(
                    d_model=args.d_model,
                    d_state=args.d_state,
                    d_conv=args.d_conv,
                    expand=args.expand,
                    dt_rank=args.dt_rank,
                    seq_len=256
                ),
                nn.LayerNorm(args.d_model),
                nn.Dropout(0.1)
            ) for _ in range(args.n_layer)
        ])
        
        # Output layers
        self.norm = nn.LayerNorm(args.d_model)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(args.d_model, num_classes)
    
    def forward(self, x):
        # Initial convolutions
        x = self.conv1(x)
        
        # Convert to sequence
        x = self.patch_to_seq(x)
        
        # Reshape to sequence
        B, C, H, W = x.shape
        x = x.view(B, C, -1)
        x = x.permute(0, 2, 1).contiguous()
        
        # Apply Mamba blocks with residual connections
        for layer in self.layers:
            x = layer(x) + x  # Explicit residual connection
        
        # Global pooling and classification
        x = x.mean(dim=1)  # Global average pooling
        x = self.norm(x)
        x = self.drop(x)
        logits = self.fc(x)
        probabilities = F.softmax(logits, dim=-1)
        
        return logits, probabilities