import sys
sys.path.insert(0, r'c:\Users\marcozhu\Desktop\6980')
import torch, torch.nn as nn

class Swish(nn.Module):
    def forward(self, x): return torch.nn.functional.silu(x)

class ResBlock(nn.Module):
    def __init__(self, dim, drop=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
            Swish(), nn.Dropout(drop),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = Swish()
        self.drop = nn.Dropout(drop)
    def forward(self, x): return self.drop(self.act(x + self.net(x)))

class FeatInteraction(nn.Module):
    def __init__(self, idim, inter_dim=64):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(idim, inter_dim), nn.BatchNorm1d(inter_dim))
        self.factors = nn.Parameter(torch.randn(idim, inter_dim) * 0.01)
        self.obn = nn.BatchNorm1d(inter_dim)
    def forward(self, x): return self.obn(self.proj(x) + torch.matmul(x, self.factors))

class MLPV2(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, num_residual_blocks=1, dropout=0.25,
                 use_fi=True, fi_dim=64, **kwargs):
        super().__init__()
        hd = hidden_dims or [128, 64]
        if 'n_res_blocks' in kwargs:
            n_res = kwargs.pop('n_res_blocks')
        else:
            n_res = num_residual_blocks
        print(f"  MLPV2 init: input_dim={input_dim}, hd={hd}, n_res={n_res}, dropout={dropout}")
        # ... rest of code

# Test both parameter names
print("Test 1: num_residual_blocks")
m1 = MLPV2(10, hidden_dims=[16, 8], num_residual_blocks=2)
print("Test 2: n_res_blocks (via kwargs)")
m2 = MLPV2(10, hidden_dims=[16, 8], n_res_blocks=3)
print("Both tests passed!")
