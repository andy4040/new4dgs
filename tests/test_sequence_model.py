import copy
import io

import pytest
import torch

from appearance import SequenceAppearance, sh_basis
from covariance_renderer import render
from geometry import Gaussians
from sequence_model import SequenceModel, load_sequence, preparation_settings


def scene():
    xyz = torch.tensor([[-.2, 0., 2.], [.2, .1, 2.2], [0., -.2, 2.1]], dtype=torch.float64)
    g = Gaussians(xyz, torch.full_like(xyz, .5), .15).double()
    cfg = dict(velocity=dict(width=8, depth=1, spatial_bands=1, time_bands=1),
               ode_method='rk4', ode_step=.5, flow_chunk_size=2,
               velocity_lr=.001, appearance_lr=.01, frame_start=0,
               train_frames=[0, 2], heldout_frames=[1],
               appearance=dict(sh_degree=2, time_rank=2, time_logit_limit=.1),
               geometry_preparation=dict(steps=6, frames=[0, 2], position_lr=.001,
                    shape_lr=.001, densify_interval=2, densify_until=2,
                    split_fraction=.5, max_gaussians=6))
    model = SequenceModel(cfg, g.state_dict(), 1.)
    return model, cfg


def camera(center=0.):
    return dict(R=torch.eye(3, dtype=torch.float64),
                t=torch.tensor([-center, 0., 0.], dtype=torch.float64),
                K=torch.tensor([[12., 0., 6.], [0., 12., 6.], [0., 0., 1.]], dtype=torch.float64),
                height=12, width=12)


def test_zero_appearance_reproduces_static_render():
    model, _ = scene()
    g = model.g
    for t in (0., .5, 1.):
        for cam in (camera(-.3), camera(.3)):
            x, c = model.transport(t, False)
            before = render(x, c, g.color(), g.opacity(), cam)['rgb']
            after = model.render(x, c, t, cam)['rgb']
            torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_sh_and_time_change_color_with_bounded_anchored_time():
    model, _ = scene()
    g, a = model.g, model.appearance
    with torch.no_grad():
        a.sh[:, 2, :] = .7
    left = a(g, g.xyz, .5, camera(-1.))
    right = a(g, g.xyz, .5, camera(1.))
    assert (left-right).abs().max() > .02
    with torch.no_grad():
        a.sh.zero_()
        a.time.fill_(100.)
    torch.testing.assert_close(a(g, g.xyz, 0., camera()), g.color(), rtol=0, atol=0)
    for t in (0., .2, .5, 1.):
        delta = torch.logit(a(g, g.xyz, t, camera()))-g.color_logits
        assert delta.abs().max() <= .10000001
    # SH energy per degree is independent of a unit viewing direction.
    b = sh_basis(torch.randn(20, 3, dtype=torch.float64), 2)
    torch.testing.assert_close(b[:, :3].square().sum(-1), b.new_full((20,), 3/(4*torch.pi)))
    torch.testing.assert_close(b[:, 3:].square().sum(-1), b.new_full((20,), 5/(4*torch.pi)))


def test_multiframe_rgb_reaches_geometry_through_checkpointed_ode():
    model, _ = scene()
    model.freeze_geometry(False)
    model.v.requires_grad_(False)
    cam = camera()
    # Deliberately a later-time observation; no reference-frame loss in this test.
    x, c = model.transport(1., True)
    target = render(x.detach()+x.new_tensor([.1, -.03, 0]), c.detach(),
                    model.g.color().detach(), model.g.opacity().detach(), cam)['rgb']
    loss = (model.render(x, c, 1., cam)['rgb']-target).square().mean()
    loss.backward()
    expected = [p.grad.clone() for p in (model.g.xyz, model.g.log_scale, model.g.lower)]
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in expected)
    assert model.appearance.sh.grad.norm() > 0 and model.appearance.time.grad.norm() > 0
    model.zero_grad(set_to_none=True)
    x, c = model.transport(1., False)
    (model.render(x, c, 1., cam)['rgb']-target).square().mean().backward()
    for p, grad in zip((model.g.xyz, model.g.log_scale, model.g.lower), expected):
        torch.testing.assert_close(p.grad, grad, rtol=1e-8, atol=1e-10)


def test_split_inherits_appearance_and_freeze_blocks_geometry_changes():
    model, _ = scene()
    model.freeze_geometry(False)
    with torch.no_grad():
        model.appearance.sh.normal_()
        model.appearance.time.normal_()
    old = {int(i): (sh.clone(), t.clone()) for i, sh, t in
           zip(model.g.ids, model.appearance.sh, model.appearance.time)}
    event = model.split(torch.tensor([1., 3., 2.]), 1/3)
    assert len(model.g.xyz) == 4
    assert len(model.g.ids.unique()) == 4
    for i, parent, sh, t in zip(model.g.ids, model.g.parent_ids, model.appearance.sh, model.appearance.time):
        source = int(i) if int(i) in old else int(parent)
        torch.testing.assert_close(sh, old[source][0])
        torch.testing.assert_close(t, old[source][1])
    assert set(event['retired_ids']).isdisjoint(model.g.ids.tolist())
    model.freeze_geometry()
    frozen = {name: value.clone() for name, value in model.g.state_dict().items()
              if name not in ('color_logits', 'opacity_logits')}
    optimizer = model.optimizer()
    x, c = model.transport(.5)
    loss = model.render(x, c, .5, camera())['rgb'].sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for name, value in frozen.items():
        torch.testing.assert_close(model.g.state_dict()[name], value, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match='frozen'):
        model.split(torch.ones(4), .25)


def test_checkpoint_roundtrip_after_split_and_reference_factor_reload():
    model, cfg = scene()
    model.freeze_geometry(False)
    model.split(torch.ones(3), 1/3)
    with torch.no_grad():
        model.appearance.sh.normal_(std=.1)
        model.appearance.time.fill_(.3)
        model.g.log_scale.add_(.2)
    state = dict(model.checkpoint(), config=cfg)
    file = io.BytesIO()
    torch.save(state, file)
    file.seek(0)
    restored = load_sequence(cfg, torch.load(file, weights_only=True), 1.)
    for t in (0., .7, 1.):
        a, b = model.transport(t, False)
        x, c = restored.transport(t, False)
        torch.testing.assert_close(a, x, rtol=0, atol=0)
        torch.testing.assert_close(b, c, rtol=0, atol=0)
        torch.testing.assert_close(model.render(a, b, t, camera(.4))['rgb'],
                                   restored.render(x, c, t, camera(.4))['rgb'], rtol=0, atol=0)


@pytest.mark.parametrize('frames', [[0, 1], [2], [0, 0], [0, 3]])
def test_preparation_rejects_holdouts_and_invalid_frame_sets(frames):
    _, cfg = scene()
    cfg['geometry_preparation']['frames'] = frames
    with pytest.raises(ValueError):
        preparation_settings(cfg)


def test_legacy_full_sequence_without_new_options_loads():
    model, cfg = scene()
    cfg = copy.deepcopy(cfg)
    del cfg['appearance']
    del cfg['geometry_preparation']
    state = {'reference': model.g.state_dict(), 'velocity': model.v.state_dict()}
    restored = load_sequence(cfg, state, 1.)
    assert list(restored.appearance.parameters()) == []
    x, c = restored.transport(.5, False)
    torch.testing.assert_close(restored.render(x, c, .5, camera())['rgb'],
                               render(x, c, restored.g.color(), restored.g.opacity(), camera())['rgb'])


def test_appearance_fits_multiple_views_and_times_with_frozen_geometry():
    model, _ = scene()
    a = model.appearance
    truth = SequenceAppearance(3, sh_degree=2, time_rank=2).double()
    with torch.no_grad():
        truth.sh[:, 2, :] = .8
        truth.time[:, 0, :] = .6
    observations = [(t, camera(c)) for t in (0., .5, 1.) for c in (-1., 1.)]
    targets = [truth(model.g, model.g.xyz, t, cam).detach() for t, cam in observations]
    opt = torch.optim.Adam(a.parameters(), lr=.05)

    def loss():
        return sum((a(model.g, model.g.xyz, t, cam)-target).square().mean()
                   for (t, cam), target in zip(observations, targets))

    initial = float(loss().detach())
    for _ in range(100):
        opt.zero_grad()
        value = loss()
        value.backward()
        opt.step()
    assert float(loss().detach()) < initial*.02
