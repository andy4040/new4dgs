"""World-space SH color residuals and bounded, reference-anchored time color.

The DC color and opacity remain in Gaussians. Evaluating SH in Python gives the
reference and CUDA renderers exactly the same per-view colors, without changing
the covariance rasterization path. Zero residuals reproduce old checkpoints.
"""
import torch
from torch import nn
from torch.nn import functional as F


def sh_basis(direction, degree):
    """Real SH, degrees 1/2, with the constant basis omitted (RGB owns DC)."""
    if degree not in (0, 1, 2):
        raise ValueError('sh_degree must be 0, 1 or 2')
    x, y, z = F.normalize(direction, dim=-1, eps=1e-8).unbind(-1)
    if degree == 0:
        return direction.new_empty((*direction.shape[:-1], 0))
    values = [-0.4886025119029199*y, 0.4886025119029199*z,
              -0.4886025119029199*x]
    if degree == 2:
        values += [1.0925484305920792*x*y, -1.0925484305920792*y*z,
                   0.31539156525252005*(3*z*z-1),
                   -1.0925484305920792*x*z, 0.5462742152960396*(x*x-y*y)]
    return torch.stack(values, dim=-1)


class SequenceAppearance(nn.Module):
    def __init__(self, count, sh_degree=0, time_rank=0, time_logit_limit=.1,
                 sh_weight=1e-4, time_weight=.01):
        super().__init__()
        if sh_degree not in (0, 1, 2) or time_rank not in (0, 1, 2):
            raise ValueError('Supported SH degrees: 0/1/2; time ranks: 0/1/2')
        if time_logit_limit <= 0 or sh_weight < 0 or time_weight < 0:
            raise ValueError('Invalid appearance bounds/regularization')
        self.sh_degree, self.time_rank = sh_degree, time_rank
        self.time_logit_limit = time_logit_limit
        self.sh_weight, self.time_weight = sh_weight, time_weight
        self.sh = nn.Parameter(torch.zeros(count, (sh_degree+1)**2-1, 3)) if sh_degree else None
        self.time = nn.Parameter(torch.zeros(count, time_rank, 3)) if time_rank else None

    def forward(self, gaussians, positions, t, camera):
        logits = gaussians.color_logits
        if self.sh is not None:
            # Direction from Gaussian to camera, in the fixed world coordinate frame.
            center = -camera['R'].T @ camera['t']
            basis = sh_basis(center-positions, self.sh_degree)
            logits = logits + (self.sh*basis[..., None]).sum(1)
        if self.time is not None:
            t = torch.as_tensor(t, device=logits.device, dtype=logits.dtype)
            basis = torch.stack([t, t*(1-t)][:self.time_rank])
            # Bound the TOTAL temporal logit change, and force it to zero at t=0.
            logits = logits + self.time_logit_limit*torch.tanh((self.time*basis[None, :, None]).sum(1))
        return logits.sigmoid()

    def regularization(self, like):
        loss = like.new_zeros(())
        if self.sh is not None:
            loss = loss + self.sh_weight*self.sh.square().mean()
        if self.time is not None:
            loss = loss + self.time_weight*self.time.square().mean()
        return loss

    @torch.no_grad()
    def remap(self, source_indices):
        """Carry every survivor/child's appearance through geometry splitting."""
        result = SequenceAppearance(len(source_indices), self.sh_degree, self.time_rank,
                                    self.time_logit_limit, self.sh_weight, self.time_weight)
        for name in ('sh', 'time'):
            value = getattr(self, name)
            if value is not None:
                setattr(result, name, nn.Parameter(value[source_indices].clone()))
        return result
