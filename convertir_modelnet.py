"""Convierte ModelNet40 original (archivos OFF) a HDF5 de 2048 puntos con el mismo formato que usa data.py.

Cada nube se obtiene muestreando 2048 puntos uniformes sobre la superficie de la malla (los triángulos
se eligen con probabilidad proporcional a su área). La semilla depende del nombre del archivo, así que
el resultado es siempre el mismo si se repite.

Orientación: los modelos originales de ModelNet40 tienen el eje vertical en z, mientras que la versión HDF5 de PointNet y
ScanObjectNN usan y como vertical. Por defecto se intercambian los ejes y y z de cada nube, (x, y, z) -> (x, z, y), para
que ModelNet40 y ScanObjectNN estén en el mismo sistema de coordenadas (lo contrario hunde la transferencia de E3).
Con --sin_rotar se deja la orientación original.

Uso:  python convertir_modelnet.py --entrada C:\\datos_modelnet\\ModelNet40 --salida datos\\modelnet40_original_2048
Salida: shape_names.txt, ply_data_train0.h5.., ply_data_test0.h5.. y muestreo.json (registro de lo que se hizo)."""
import argparse, glob, hashlib, json, os, sys, zlib
from multiprocessing import Pool
import numpy as np
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("--entrada", required=True, help="carpeta ModelNet40 con una subcarpeta por clase")
ap.add_argument("--salida", required=True)
ap.add_argument("--puntos", type=int, default=2048)
ap.add_argument("--por_archivo", type=int, default=2048, help="muestras por archivo .h5")
ap.add_argument("--sin_rotar", action="store_true", help="no intercambiar los ejes y, z")
ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
a = ap.parse_args()


def leer_off(path):
    """Lee vértices y caras de un OFF. Maneja el caso de ModelNet donde 'OFF' y los conteos van pegados ('OFF490 518 0')."""
    with open(path, "r", errors="ignore") as f:
        txt = f.read().split()
    if txt[0].startswith("OFF"):
        resto = txt[0][3:]
        txt = ([resto] if resto else []) + txt[1:]
    nv, nf = int(txt[0]), int(txt[1])
    v = np.array(txt[3:3 + 3 * nv], dtype=np.float64).reshape(nv, 3)
    i, caras = 3 + 3 * nv, []
    for _ in range(nf):
        k = int(txt[i]); idx = [int(x) for x in txt[i + 1:i + 1 + k]]; i += 1 + k
        for j in range(1, k - 1): caras.append((idx[0], idx[j], idx[j + 1]))   # triangula polígonos en abanico
    return v, np.array(caras, dtype=np.int64)


def muestrear(path):
    v, c = leer_off(path)
    a_, b_, c_ = v[c[:, 0]], v[c[:, 1]], v[c[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b_ - a_, c_ - a_), axis=1)
    if not np.isfinite(area).all() or area.sum() <= 0: raise ValueError("malla sin área")
    rng = np.random.default_rng(zlib.crc32(os.path.basename(path).encode()))
    t = rng.choice(len(c), size=a.puntos, p=area / area.sum())
    r1, r2 = rng.random(a.puntos), rng.random(a.puntos); s = np.sqrt(r1)
    p = (1 - s)[:, None] * a_[t] + (s * (1 - r2))[:, None] * b_[t] + (s * r2)[:, None] * c_[t]
    p = p.astype(np.float32)
    return p if a.sin_rotar else p[:, [0, 2, 1]]   # z vertical (OFF) -> y vertical (HDF5 de PointNet y ScanObjectNN)


def trabajo(args):
    path, lab = args
    try: return muestrear(path), lab, None
    except Exception as e: return None, lab, f"{path}: {e}"


if __name__ == "__main__":
    clases = sorted(d for d in os.listdir(a.entrada) if os.path.isdir(os.path.join(a.entrada, d)))
    assert len(clases) == 40, f"Se esperaban 40 clases y hay {len(clases)} en {a.entrada}"
    os.makedirs(a.salida, exist_ok=True)
    open(os.path.join(a.salida, "shape_names.txt"), "w").write("\n".join(clases) + "\n")
    reg = {"clases": clases, "puntos": a.puntos, "metodo": "uniforme sobre la superficie (triangulos ponderados por area)",
           "ejes": "originales (z vertical)" if a.sin_rotar else "intercambiados y<->z (y vertical, como el HDF5 de PointNet y ScanObjectNN)", "fallidos": []}
    for split in ("train", "test"):
        lista = [(p, k) for k, c in enumerate(clases) for p in sorted(glob.glob(os.path.join(a.entrada, c, split, "*.off")))]
        print(f"{split}: {len(lista)} modelos", flush=True)
        with Pool(a.procesos) as pool:
            res = []
            for n, r in enumerate(pool.imap(trabajo, lista, chunksize=16), 1):
                res.append(r)
                if n % 1000 == 0: print(f"  {n}/{len(lista)}", flush=True)
        ok = [(x, l) for x, l, e in res if e is None]; reg["fallidos"] += [e for _, _, e in res if e]
        reg[f"{split}_modelos"] = len(ok)
        X, y = np.stack([x for x, _ in ok]), np.array([l for _, l in ok], dtype=np.int64)
        for j, ini in enumerate(range(0, len(y), a.por_archivo)):
            with h5py.File(os.path.join(a.salida, f"ply_data_{split}{j}.h5"), "w") as f:
                f["data"] = X[ini:ini + a.por_archivo]; f["label"] = y[ini:ini + a.por_archivo].reshape(-1, 1)
    json.dump(reg, open(os.path.join(a.salida, "muestreo.json"), "w"), indent=1, ensure_ascii=False)
    print("Listo.", {k: v for k, v in reg.items() if k.endswith("_modelos")}, "| fallidos:", len(reg["fallidos"]))
    if reg["fallidos"]: print("Primeros fallos:", *reg["fallidos"][:5], sep="\n  ")
