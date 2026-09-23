import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from data import N3DV


def fixture_dataset(root):
    p=np.zeros((3,3,5));p[:,:,:3]=np.eye(3);p[:,:,4]=[24,32,30];p[:,0,3]=[-1,0,1]
    np.save(root/'poses_bounds.npy',np.concatenate([p.reshape(3,15),np.ones((3,2))],axis=1))
    for c in range(3):
        d=root/f'cam{c:02d}'/'images';d.mkdir(parents=True)
        for f in range(31):Image.fromarray(np.zeros((24,32,3),dtype=np.uint8)).save(d/f'{f:04d}.png')


def test_strict_access_and_camera_count(tmp_path):
    fixture_dataset(tmp_path)
    cfg=json.loads((Path(__file__).parents[1]/'configs/smoke.json').read_text());cfg.update(data_root=str(tmp_path),image_width=32)
    d=N3DV(cfg)
    assert d.train_cameras==['cam01','cam02']
    assert len(d.keys('train'))==32 and len(d.keys('view'))==16 and len(d.keys('time'))==10
    for c,f in [('cam00',10),('cam01',12),('cam02',28)]:
        with pytest.raises(PermissionError):d.image(c,f,'train')
    assert len(d.audit)==0
    assert d.image('cam01',10).shape==(24,32,3)


def test_missing_camera_id_uses_sorted_calibration_rows(tmp_path):
    fixture_dataset(tmp_path)
    (tmp_path/'cam02').rename(tmp_path/'cam05')
    cfg=json.loads((Path(__file__).parents[1]/'configs/smoke.json').read_text())
    cfg.update(data_root=str(tmp_path),image_width=32)
    d=N3DV(cfg)
    assert d.names==['cam00','cam01','cam05']
    assert d.inventory['cam05']['calibration_row']==2
    assert d.cameras['cam05']['center']==[1.,0.,0.]
    with pytest.raises(PermissionError):d.image('cam05',12,'train')
