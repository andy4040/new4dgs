"""Read-only N3DV loader, matching hustvl/4DGaussians camera transforms.
Images are accessed only through split-specific views. Frames are zero based.
"""
from pathlib import Path
import json
import re
import cv2
import numpy as np
import torch
from PIL import Image


def validate_split(cfg):
    a, b = cfg['train_frames'], cfg['heldout_frames']
    assert len(a) == len(set(a)) == 16 and len(b) == len(set(b)) == 5
    assert not set(a) & set(b) and set(a + b) == set(range(10, 31))
    assert 10 in a and 30 in a and cfg['test_camera'] == 'cam00'
    assert cfg['frame_index_origin'] == 0


class N3DV:
    def __init__(self, cfg):
        validate_split(cfg)
        if not cfg.get('data_root'):
            raise FileNotFoundError('Set --data to external coffee_martini directory; data is not bundled.')
        self.cfg, self.root = cfg, Path(cfg['data_root']).expanduser().resolve()
        raw = np.load(self.root / 'poses_bounds.npy')
        poses = raw[:, :-2].reshape(-1, 3, 5)
        names = {p.stem for p in self.root.glob('cam*.mp4')}
        names |= {p.name for p in self.root.glob('cam*') if p.is_dir()}
        self.names = sorted(n for n in names if re.fullmatch(r'cam\d+', n))
        # Official N3DV: calibration rows follow sorted existing streams; gaps are valid.
        if len(self.names) != len(poses) or len({int(n[3:]) for n in self.names}) != len(self.names):
            raise ValueError('Camera names must map unambiguously to calibration rows: ' + str(self.names))
        if len(self.names) < 3 or cfg['test_camera'] not in self.names:
            raise ValueError('Need cam00 and at least two training cameras')
        self.train_cameras = [n for n in self.names if n != cfg['test_camera']]
        # Exact upstream conversion: [col1,-col0,col2,translation], then flip y/z.
        p = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        self.cameras = {}
        self.audit = []
        self.cache = {}
        self.inventory = {}
        for i, name in enumerate(self.names):
            h, w, f = poses[i, :, -1]
            width = cfg['image_width']; height = int(round(h * width / w))
            R = p[i, :3, :3].copy(); R[:, 1:] *= -1
            t = -p[i, :3, 3] @ R
            K = np.array([[f*width/w,0,width/2],[0,f*height/h,height/2],[0,0,1.]])
            self.cameras[name] = {'R': torch.tensor(R.T, dtype=torch.float32),
                't': torch.tensor(t, dtype=torch.float32), 'K': torch.tensor(K,dtype=torch.float32),
                'width': width, 'height': height, 'center': p[i,:3,3].tolist()}
            image_dir = self.root / name / 'images'
            files = sorted(image_dir.glob('*.png')) + sorted(image_dir.glob('*.jpg'))
            movie = self.root / (name+'.mp4')
            cap = cv2.VideoCapture(str(movie)) if movie.exists() else None
            fps = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap is not None else len(files)
            if cap is not None: cap.release()
            self.inventory[name] = {'images': len(files), 'video_frames': count, 'fps': fps, 'calibration_row': i, 'calibration_height':float(h), 'calibration_width':float(w), 'calibration_focal':float(f)}
            if not files and count < 31: raise ValueError(f'{name}: fewer than 31 frames')
        centers = np.array([self.cameras[n]['center'] for n in self.train_cameras])
        self.extent = float(np.linalg.norm(centers-centers.mean(0), axis=1).max()*1.1)
        if self.extent <= 0: raise ValueError('Degenerate camera calibration')

    def keys(self, split):
        if split == 'train': cams, frames = self.train_cameras, self.cfg['train_frames']
        elif split == 'view': cams, frames = ['cam00'], self.cfg['train_frames']
        elif split == 'time': cams, frames = self.train_cameras, self.cfg['heldout_frames']
        elif split == 'joint': cams, frames = ['cam00'], self.cfg['heldout_frames']
        else: raise ValueError(split)
        return [(c,f) for f in frames for c in cams]

    def image(self, camera, frame, split='train'):
        if (camera,frame) not in self.keys(split):
            raise PermissionError(f'Forbidden {split} access: {camera}/{frame}')
        self.audit.append({'split':split,'camera':camera,'frame':frame})
        key = (camera, frame)
        if key not in self.cache:
            folder = self.root / camera / 'images'
            candidates = [p for p in folder.glob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg') and p.stem.isdigit() and int(p.stem)==frame]
            if len(candidates) > 1: raise ValueError('Duplicate image frame')
            if candidates:
                # Upstream frame naming starts at 0000; reject ambiguous one-based folders.
                if not any(p.stem.isdigit() and int(p.stem)==0 for p in folder.glob('*')):
                    raise ValueError('Image folder has no frame 0; confirm numbering before use')
                img = np.array(Image.open(candidates[0]).convert('RGB'))
            else:
                cap = cv2.VideoCapture(str(self.root/(camera+'.mp4')))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame); ok, img = cap.read(); cap.release()
                if not ok: raise FileNotFoundError(f'{camera}/{frame}')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            c = self.cameras[camera]
            img = cv2.resize(img, (c['width'], c['height']), interpolation=cv2.INTER_AREA)
            self.cache[key] = torch.from_numpy(img.copy()).float()/255
        return self.cache[key]

    def camera(self, name, device):
        return {**{k:v.to(device) if isinstance(v, torch.Tensor) else v for k,v in self.cameras[name].items()}, 'renderer':self.cfg.get('renderer','reference')}

    def save_audit(self, path):
        Path(path).write_text(json.dumps({'inventory':self.inventory,'accesses':self.audit},indent=2))


def normalized_time(frame):
    return (frame - 10) / 20.
