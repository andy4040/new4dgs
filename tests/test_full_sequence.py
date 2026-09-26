"""Small CPU scene exercises real preparation/training/reload, not a mock renderer."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import full_sequence
from covariance_renderer import render
from geometry import Gaussians


def make_fixture(root):
    data_root = root/'data'
    data_root.mkdir()
    poses = []
    xyz = torch.tensor([[-.18, 0., 2.], [.18, .12, 2.2], [0., -.18, 2.1]], dtype=torch.float32)
    rgb = torch.tensor([[.7, .3, .2], [.2, .6, .3], [.3, .4, .7]], dtype=torch.float32)
    g = Gaussians(xyz, rgb, .13)
    with torch.no_grad():
        g.opacity_logits.fill_(1.)
    for i, center in enumerate((0., -.3, .3)):
        pose = np.zeros((3, 5))
        pose[:, :3] = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
        pose[:, 3] = [center, 0, 0]
        pose[:, 4] = [12, 12, 12]
        poses.append(pose)
        folder = data_root/f'cam{i:02d}'/'images'
        folder.mkdir(parents=True)
        cam = dict(R=torch.eye(3, dtype=torch.float32), t=torch.tensor([-center, 0., 0.], dtype=torch.float32),
                   K=torch.tensor([[12., 0, 6], [0, 12., 6], [0, 0, 1.]], dtype=torch.float32), height=12, width=12)
        for frame in range(4):
            with torch.no_grad():
                # Later frames differ in position and appearance; first-frame geometry is imperfect.
                x = xyz+xyz.new_tensor([.08*frame/3, 0., 0.])
                color = (g.color_logits+.05*frame/3+.1*center).sigmoid()
                image = render(x, g.covariance(), color, g.opacity(), cam)['rgb']
            Image.fromarray((image.numpy()*255).astype('uint8')).save(folder/f'{frame:04d}.png')
    np.save(data_root/'poses_bounds.npy', np.concatenate([np.array(poses).reshape(3, 15), np.ones((3, 2))], 1))
    reference = root/'reference.pt'
    torch.save(dict(reference=g.state_dict(), config=dict(frame_start=0)), reference)
    cfg = dict(data_root=str(data_root), seed=7, device='cpu', frame_start=0, frame_end=3,
               train_frames=[0, 2, 3], heldout_frames=[1], test_camera='cam00', frame_index_origin=0,
               image_width=12, renderer='reference', reference_checkpoint=str(reference), reference_min_psnr=0.,
               velocity=dict(width=4, depth=1, spatial_bands=1, time_bands=1),
               ode_method='rk4', ode_step=.5, flow_chunk_size=2, velocity_lr=.001, appearance_lr=.01,
               views_per_step=2, gradient_clip=1., acceleration_weight=1e-6,
               trajectory_steps=4, checkpoint_interval=1, log_interval=2,
               monitor_frames=[0, 2, 3], monitor_cameras=['cam01', 'cam02'], preview_frames=[],
               early_stopping=dict(enabled=True, eval_interval=2, patience=5, min_delta=.0001, min_steps=100),
               appearance=dict(sh_degree=2, time_rank=2, time_logit_limit=.1, sh_weight=1e-4, time_weight=.01),
               geometry_preparation=dict(steps=6, frames=[0, 2, 3], position_lr=.002, shape_lr=.001,
                    densify_interval=3, densify_until=3, split_fraction=.5, max_gaussians=4))
    path = root/'config.json'
    path.write_text(json.dumps(cfg))
    return path, cfg


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def test_full_training_freezes_prepared_geometry_and_reloads_exact_evaluation(tmp_path):
    config, cfg = make_fixture(tmp_path)
    out = tmp_path/'run'
    full_sequence.main(SimpleNamespace(config=str(config), out=str(out), resume=None))
    preparation = load(out/'preparation_latest.pt')
    motion = load(out/'latest.pt')
    assert preparation['phase'] == 'preparation' and motion['phase'] == 'motion'
    assert len(motion['reference']['xyz']) == 4  # Actual split, respecting cap.
    assert preparation['lineage'] and set(h['frame'] for h in preparation['history']) == {0, 2, 3}
    for key in ('xyz', 'log_scale', 'lower', 'ids', 'parent_ids'):
        torch.testing.assert_close(motion['reference'][key], preparation['reference'][key], rtol=0, atol=0)
    # A surviving original point was corrected using the multi-frame objective.
    reference = load(cfg['reference_checkpoint'])['reference']
    for row, point_id in enumerate(preparation['reference']['ids']):
        if point_id < len(reference['xyz']):
            assert not torch.equal(preparation['reference']['xyz'][row], reference['xyz'][point_id])
    assert motion['sequence_appearance']['sh'].abs().max() > 0
    assert motion['sequence_appearance']['time'].abs().max() > 0
    for filename in ('preparation_access.json', 'training_access.json'):
        accesses = json.loads((out/filename).read_text())['accesses']
        assert accesses and all(r['split'] == 'train' and r['camera'] != 'cam00' and
                                r['frame'] in cfg['train_frames'] for r in accesses)
    import evaluate
    reloaded = tmp_path/'reloaded'
    evaluate.main(SimpleNamespace(checkpoint=str(out/'best.pt'), data=None, out=str(reloaded), device='cpu'))
    before = json.loads((out/'evaluation.json').read_text())
    after = json.loads((reloaded/'evaluation.json').read_text())
    assert before['image'] == after['image']
    assert before['per_image'] == after['per_image']
    assert {r['frame'] for r in after['per_image']} == {0, 1, 2, 3}
    assert (out/'cam00_target_render.mp4').stat().st_size > 0


@pytest.mark.parametrize('phase,step', [('preparation', 3), ('motion', 2)])
def test_interrupted_resume_matches_uninterrupted_parameters_and_selection(tmp_path, monkeypatch, phase, step):
    config, _ = make_fixture(tmp_path)
    uninterrupted, interrupted = tmp_path/'continuous', tmp_path/'interrupted'
    # Evaluation itself is covered above; avoid repeated video/plot creation here.
    monkeypatch.setattr(full_sequence, 'evaluate_all', lambda *args: None)
    full_sequence.main(SimpleNamespace(config=str(config), out=str(uninterrupted), resume=None))
    original_save = full_sequence.save
    filename = 'preparation_latest.pt' if phase == 'preparation' else 'latest.pt'

    class Interrupted(Exception):
        pass

    def stop_after_saved(path, state):
        original_save(path, state)
        if path.name == filename and state['phase'] == phase and state['step'] == step:
            raise Interrupted()

    monkeypatch.setattr(full_sequence, 'save', stop_after_saved)
    with pytest.raises(Interrupted):
        full_sequence.main(SimpleNamespace(config=str(config), out=str(interrupted), resume=None))
    monkeypatch.setattr(full_sequence, 'save', original_save)
    full_sequence.main(SimpleNamespace(config=str(config), out=str(interrupted), resume=str(interrupted/filename)))
    expected, actual = load(uninterrupted/'latest.pt'), load(interrupted/'latest.pt')
    for group in ('reference', 'velocity', 'sequence_appearance'):
        for key in expected[group]:
            torch.testing.assert_close(actual[group][key], expected[group][key], rtol=0, atol=0)
    assert actual['monitor'] == expected['monitor']
    assert actual['stopper'] == expected['stopper']
    assert load(uninterrupted/'best.pt')['step'] == load(interrupted/'best.pt')['step']
