import torch 
import torch.nn as nn 

class RepaMLP(nn.Module):
    def __init__(self, hidden_size=1152, projector_dim=2048, z_dim=1536):
        super().__init__()
        self.proj = nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )
       

        def forward(self, x):
            return self.proj(x)