"""Corre los experimentos en un PC con GPU (Windows/Linux). Es el mismo flujo del cuaderno de Colab.
Uso (desde esta carpeta):   python run_local.py
Si se corta o se cierra la ventana, se vuelve a correr el mismo comando y sigue donde quedó.
Opciones:  --semillas 1 2 3   --modelos pointnet dgcnn   --tareas mn40 sonn mn2sonn   --epocas 100
           --minutos 0 (sin límite)   --bs_dgcnn 32   --amp_dgcnn 1"""
import argparse, glob, json, os, shutil, subprocess, sys, time, urllib.request, zipfile

ap = argparse.ArgumentParser()
ap.add_argument("--semillas", type=int, nargs="+", default=[1, 2, 3])
ap.add_argument("--modelos", nargs="+", default=["pointnet", "dgcnn"])
ap.add_argument("--tareas", nargs="+", default=["mn40", "sonn", "mn2sonn"])
ap.add_argument("--epocas", type=int, default=100)
ap.add_argument("--minutos", type=float, default=0)
ap.add_argument("--amp_dgcnn", type=int, default=1)
ap.add_argument("--bs_dgcnn", type=int, default=32)
ap.add_argument("--datos", default="datos"); ap.add_argument("--res", default="results")
ap.add_argument("--mn40"); ap.add_argument("--sonn")
a = ap.parse_args()
AMP = {"pointnet": 0, "dgcnn": a.amp_dgcnn}; BS = {"pointnet": 32, "dgcnn": a.bs_dgcnn}; CADA = {"pointnet": 5, "dgcnn": 1}
os.makedirs(a.res, exist_ok=True); os.makedirs(os.path.join(a.res, "avance"), exist_ok=True)

# ---- GPU ----
import torch
print("PyTorch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO HAY GPU")
if not torch.cuda.is_available():
    sys.exit("No se detectó GPU con CUDA. Instalar PyTorch con CUDA desde https://pytorch.org/get-started/locally/")

# ---- datos (se bajan una sola vez) ----
def bajar(url, dest):
    if os.path.exists(dest): return
    print("Descargando", os.path.basename(dest), "...", flush=True)
    urllib.request.urlretrieve(url, dest + ".part"); os.replace(dest + ".part", dest)
MN40, SONN = a.mn40, a.sonn
if not MN40:
    os.makedirs(a.datos, exist_ok=True); f = glob.glob(f"{a.datos}/**/shape_names.txt", recursive=True)
    if not f:
        z = f"{a.datos}/mn40.zip"; bajar("https://huggingface.co/datasets/Msun/modelnet40/resolve/main/modelnet40_ply_hdf5_2048.zip", z)
        zipfile.ZipFile(z).extractall(a.datos); f = glob.glob(f"{a.datos}/**/shape_names.txt", recursive=True)
    MN40 = os.path.dirname(f[0])
if not SONN:
    SONN = f"{a.datos}/h5_files"; os.makedirs(f"{SONN}/main_split", exist_ok=True)
    for n in ("training", "test"):
        fn = f"{n}_objectdataset_augmentedrot_scale75.h5"
        bajar(f"https://huggingface.co/datasets/cminst/ScanObjectNN/resolve/main/main_split/{fn}", f"{SONN}/main_split/{fn}")
print("ModelNet40:", MN40, "| ScanObjectNN:", SONN)

# ---- reutiliza corridas viejas de PointNet (semilla 1) solo si la configuración coincide ----
for t in a.tareas:
    old, new = f"{a.res}/{t}_pointnet.json", f"{a.res}/{t}_pointnet_s1.json"
    if os.path.exists(old) and not os.path.exists(new):
        g = json.load(open(old))["args"]
        if g["seed"] == 1 and g["epochs"] == a.epocas and g["npoints"] == 1024 and g["bs"] == 32:
            shutil.copy(old, new); shutil.copy(f"{a.res}/{t}_pointnet_samples.npz", f"{a.res}/{t}_pointnet_s1_samples.npz")
            print("Reutilizado:", t, "pointnet semilla 1")

# ---- entrenamientos ----
t_ini, pausado = time.time(), False
listo = lambda t, m, s: os.path.exists(f"{a.res}/{t}_{m}_s{s}.json")
for s in a.semillas:
    for m in a.modelos:
        for t in a.tareas:
            if listo(t, m, s): continue
            lim = ""
            if a.minutos:
                rest = a.minutos - (time.time() - t_ini) / 60
                if rest <= 0.5: pausado = True; continue
                lim = f"--stop_after_min {rest:.2f}"
            print(f"\n=== {t} | {m} | semilla {s} | {a.epocas} épocas ===", flush=True)
            ck = f"{a.res}/avance/ckpt_{t}_{m}_s{s}_e{a.epocas}.pt"
            cmd = [sys.executable, "-u", "train.py", "--task", t, "--model", m, "--seed", str(s), "--epochs", str(a.epocas),
                   "--amp", str(AMP[m]), "--bs", str(BS[m]), "--mn40", MN40, "--sonn", SONN, "--out", a.res,
                   "--ckpt", ck, "--ckpt_every", str(CADA[m]),
                   "--workers", "0" if os.name == "nt" else "2"] + lim.split()   # si train.py es viejo, falla al instante en vez de colgarse
            rc = subprocess.run(cmd).returncode
            if rc == 3: pausado = True
            elif rc != 0: sys.exit(f"Falló {t} {m} semilla {s} (código {rc}). Ver el mensaje de arriba.")
hechos = sum(listo(t, m, s) for s in a.semillas for m in a.modelos for t in a.tareas)
print(f"\nTerminados: {hechos}/{len(a.semillas)*len(a.modelos)*len(a.tareas)}")
if pausado: sys.exit("Pausado por tiempo. Vuelve a correr el mismo comando para continuar.")
subprocess.run([sys.executable, "aggregate.py", "--res", a.res, "--out", "figs"])
print("Listo. Mándame la carpeta results (los .json, sin la subcarpeta avance), summary.json y figs.")
