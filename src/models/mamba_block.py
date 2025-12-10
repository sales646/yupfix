"""
Mamba-2 Selective State Space Model Block
Pure PyTorch implementation (no CUDA dependencies)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MambaBlock(nn.Module):
    """
    Single Mamba block implementing selective state space model.
    
    Key features:
    - Input-dependent gating (selectivity)
    - Linear complexity O(n)
    - Causal (no future leakage)
    """
    def __init__(self, d_model: int, d_state: int = 64, expand: int = 2, 
                 d_conv: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.d_conv = d_conv
        
        # Input projection (to 2x for gating)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution (causal)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, 
            kernel_size=d_conv, 
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, dt
        
        # A is log-spaced (learnable)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (Batch, SeqLen, d_model)
        Returns:
            (Batch, SeqLen, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Input projection and split for gating
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)
        
        # Causal convolution
        x_conv = x_in.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Causal: trim future
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)
        
        # SSM parameters from input
        x_ssm = self.x_proj(x_conv)  # (B, L, 2*d_state + 1)
        B = x_ssm[:, :, :self.d_state]
        C = x_ssm[:, :, self.d_state:2*self.d_state]
        dt = F.softplus(x_ssm[:, :, -1])  # (B, L)
        
        # Discretize A
        A = -torch.exp(self.A_log)  # (d_state,)
        
        # Selective scan (chunked for performance)
        y = self._selective_scan_chunked(x_conv, A, B, C, dt)
        
        # Skip connection with D
        y = y + x_conv * self.D
        
        # Gate with z
        y = y * F.silu(z)
        
        # Output projection
        y = self.out_proj(y)
        y = self.dropout(y)
        
        return y
    
    def _selective_scan_chunked(self, x, A, B, C, dt, chunk_size=1000):
        """
        Chunked selective scan for memory efficiency and performance.
        
        Args:
            x: (B, L, d_inner)
            A: (d_state,)
            B: (B, L, d_state)
            C: (B, L, d_state)
            dt: (B, L)
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[0]
        
        # Initialize state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            
            # Extract chunk
            x_chunk = x[:, chunk_start:chunk_end]
            B_chunk = B[:, chunk_start:chunk_end]
            C_chunk = C[:, chunk_start:chunk_end]
            dt_chunk = dt[:, chunk_start:chunk_end]
            
            # Process chunk
            chunk_out, h = self._process_chunk(x_chunk, h, A, B_chunk, C_chunk, dt_chunk)
            outputs.append(chunk_out)
            
        return torch.cat(outputs, dim=1)  # (B, L, d_inner)

    def _process_chunk(self, x, h, A, B, C, dt):
        """Process a single chunk sequentially"""
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[0]
        
        outputs = []
        
        # Pre-calculate A_bar and B_bar for the chunk if possible, 
        # but dt varies per timestep so we do it in loop or vectorized.
        # Vectorized is better but complex to implement purely in pytorch without custom kernel for scan.
        # We stick to sequential loop for correctness but chunking helps memory.
        
        for t in range(seq_len):
            # Discretized A for this timestep
            # A is (d_state,), dt is (B, 1)
            dt_t = dt[:, t].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
            
            # Fix A_bar calculation:
            # A.view(1, 1, -1) -> (1, 1, d_state)
            # dt_t -> (B, 1, 1)
            # Result -> (B, 1, d_state)
            A_bar = torch.exp(A.view(1, 1, -1) * dt_t) 
            
            # B for this timestep
            B_t = B[:, t, :].unsqueeze(1)  # (B, 1, d_state)
            
            # Fix B_bar discretization: Zero-Order Hold
            # B_bar = B_t * dt_t
            B_bar = B_t * dt_t # (B, 1, d_state)
            
            # Update state: h = A_bar * h + B_bar * x
            x_t = x[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)
            
            # h: (B, d_inner, d_state)
            # A_bar: (B, 1, d_state) -> broadcasts to (B, d_inner, d_state)
            # x_t: (B, d_inner, 1)
            # B_bar: (B, 1, d_state)
            # x_t * B_bar -> (B, d_inner, d_state)
            
            h = A_bar * h + x_t * B_bar
            
            # Output: y = C * h
            C_t = C[:, t, :].unsqueeze(1)  # (B, 1, d_state)
            y_t = (h * C_t).sum(dim=-1)  # (B, d_inner)
            
            outputs.append(y_t)
            
        return torch.stack(outputs, dim=1), h
