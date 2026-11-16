"""Carga de ModelNet40 (HDF5, 2048 pts) y ScanObjectNN (HDF5, 2048 pts)."""
import glob, os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

SONN_CLASSES = ["bag", "bin", "box", "cabinet", "chair", "desk", "display", "door",
                "shelf", "table", "bed", "pillow", "sink", "sofa", "toilet"]
# Clases compartidas entre ModelNet40 y ScanObjectNN (correspondencia elegida en este trabajo)
SHARED = [("bed", "bed"), ("chair", "chair"), ("desk", "desk"), ("door", "door"),
          ("bookshelf", "shelf"), ("monitor", "display"), ("sink", "sink"),
          ("sofa", "sofa"), ("table", "table"), ("toilet", "toilet")]
SHARED_NAMES = [s for _, s in SHARED]

SONN_FILES = {   # variante -> (subcarpeta, archivo base)
    "OBJ_ONLY": ("main_split_nobg", "objectdataset.h5"),
    "OBJ_BG": ("main_split", "objectdataset.h5"),
    "PB_T50_RS": ("main_split", "objectdataset_augmentedrot_scale75.h5"),
}


def _read_h5(paths):
    X, y = [], []
    for p in paths:
        with h5py.File(p, "r") as f:
            X.append(f["data"][:].astype(np.float32)); y.append(f["label"][:].reshape(-1).astype(np.int64))
    return np.concatenate(X), np.concatenate(y)


def modelnet40(root, split):
    """root = carpeta modelnet40_ply_hdf5_2048"""
    paths = sorted(glob.glob(os.path.join(root, f"ply_data_{split}*.h5")))
    assert paths, f"No hay archivos ply_data_{split}*.h5 en {root}"
    names = [l.strip() for l in open(os.path.join(root, "shape_names.txt"))]
    X, y = _read_h5(paths)
    return X, y, names


def scanobjectnn(root, split, variant="PB_T50_RS"):
    """root = carpeta h5_files"""
    sub, base = SONN_FILES[variant]
    pre = "training" if split == "train" else "test"
    p = os.path.join(root, sub, f"{pre}_{base}")
    assert os.path.exists(p), f"No existe {p}"
    X, y = _read_h5([p])
    return X, y, SONN_CLASSES


def restrict_shared(X, y, names, side):
    """Deja solo las 10 clases compartidas y reetiqueta 0..9 segun SHARED_NAMES."""
    col = 0 if side == "mn40" else 1
    keep, new = [], []
    for i, lab in enumerate(y):
        nm = names[lab]
        for j, pair in enumerate(SHARED):
            if pair[col] == nm:
                keep.append(i); new.append(j); break
    keep = np.array(keep)
    return X[keep], np.array(new, dtype=np.int64)


def normalize(X):
    X = X - X.mean(axis=1, keepdims=True)
    return X / np.maximum(np.linalg.norm(X, axis=2).max(axis=1)[:, None, None], 1e-8)


class Clouds(Dataset):
    def __init__(self, X, y, npoints=1024, train=False):
        self.X, self.y, self.n, self.train = normalize(X[:, :, :3]), y, npoints, train

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        pts = self.X[i]
        if self.train:
            pts = pts[np.random.permutation(pts.shape[0])[:self.n]]
            pts = pts * np.random.uniform(2 / 3, 3 / 2, size=(1, 3)).astype(np.float32)
            pts = pts + np.random.uniform(-0.2, 0.2, size=(1, 3)).astype(np.float32)
            pts = pts + np.clip(0.01 * np.random.randn(*pts.shape), -0.05, 0.05).astype(np.float32)
        else:
            pts = pts[:self.n]
        return torch.from_numpy(pts.astype(np.float32)), int(self.y[i])
