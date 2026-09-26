"""Shared construction, rendering and checkpoints for full-sequence models."""
import torch
from torch import nn
from appearance import SequenceAppearance
from covariance_renderer import render
from geometry import Gaussians, split_gaussians
from transport import SceneVelocity, chunked_flow


class SequenceModel(nn.Module):
    def __init__(self, cfg, reference, extent, velocity=None, appearance=None):
        super().__init__()
        self.cfg = cfg
        self.g = Gaussians(reference['xyz'], reference['color_logits'].sigmoid(), 1.).to(reference['xyz'])
        self.g.load_state_dict(reference)
        self.v = SceneVelocity(self.g.xyz.detach().median(0).values, extent,
                               **cfg['velocity']).to(self.g.xyz)
        if velocity is not None:
            self.v.load_state_dict(velocity)
        self.appearance = SequenceAppearance(len(self.g.xyz), **cfg.get('appearance', {})).to(self.g.xyz)
        if appearance is not None:
            self.appearance.load_state_dict(appearance)
        self.freeze_geometry()

    def freeze_geometry(self, frozen=True):
        for value in (self.g.xyz, self.g.log_scale, self.g.lower):
            value.requires_grad_(not frozen)

    def transport(self, t, checkpoint_grad=True):
        # Recompute the factor: it changes during preparation and after reloads.
        return chunked_flow(self.v, self.g.xyz, self.g.factor(), t,
                            self.cfg['ode_method'], self.cfg['ode_step'],
                            self.cfg['flow_chunk_size'], checkpoint_grad)

    def render(self, positions, covariance, t, camera):
        color = self.appearance(self.g, positions, t, camera)
        return render(positions, covariance, color, self.g.opacity(), camera)

    def optimizer(self, preparation=False):
        groups = [{'params': self.v.parameters(), 'lr': self.cfg['velocity_lr']},
                  {'params': [self.g.color_logits, self.g.opacity_logits,
                              *self.appearance.parameters()], 'lr': self.cfg['appearance_lr']}]
        if preparation:
            settings = self.cfg['geometry_preparation']
            groups += [{'params': [self.g.xyz], 'lr': settings['position_lr']*float(self.v.scale)},
                       {'params': [self.g.log_scale, self.g.lower], 'lr': settings['shape_lr']}]
        return torch.optim.Adam(groups)

    @torch.no_grad()
    def split(self, scores, fraction):
        if not self.g.xyz.requires_grad:
            raise RuntimeError('Gaussian IDs are frozen outside geometry preparation')
        self.g, event, sources = split_gaussians(self.g, scores, fraction, return_sources=True)
        self.appearance = self.appearance.remap(sources)
        return event

    def checkpoint(self):
        return {'model_format': 'full_sequence_v2', 'reference': self.g.state_dict(),
                'velocity': self.v.state_dict(), 'sequence_appearance': self.appearance.state_dict()}


def load_sequence(cfg, state, extent):
    if 'velocity' in state and cfg.get('appearance') and 'sequence_appearance' not in state:
        raise ValueError('Checkpoint has no sequence appearance; start a new run from the static reference')
    return SequenceModel(cfg, state['reference'], extent, state.get('velocity'),
                         state.get('sequence_appearance'))


def preparation_settings(cfg):
    settings = cfg.get('geometry_preparation', {})
    steps = settings.get('steps', 0)
    if not isinstance(steps, int) or steps < 0:
        raise ValueError('geometry_preparation.steps must be a nonnegative integer')
    if not steps:
        return settings
    frames = settings['frames']
    if (len(set(frames)) != len(frames) or len(frames) < 2 or
            not set(frames) <= set(cfg['train_frames']) or cfg['frame_start'] not in frames):
        raise ValueError('Preparation needs distinct training frames including the reference; no holdouts')
    if steps < len(frames) or settings['position_lr'] <= 0 or settings['shape_lr'] <= 0:
        raise ValueError('Preparation must cover every selected frame and have positive learning rates')
    interval, until = settings['densify_interval'], settings['densify_until']
    if not isinstance(interval, int) or interval < 1 or not 0 <= until < steps:
        raise ValueError('Densification must stop before preparation ends')
    if until and (interval % len(frames) or steps-until < len(frames)):
        raise ValueError('Split intervals must cover complete frame cycles, with a post-split refinement cycle')
    if not 0 < settings['split_fraction'] <= 1 or settings['max_gaussians'] < 1:
        raise ValueError('Invalid preparation Gaussian budget')
    return settings
