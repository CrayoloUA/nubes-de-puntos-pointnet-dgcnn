
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix
import data as D
from models import build

p = argparse.ArgumentParser()
p.add_argument("--task", choices=["mn40", "sonn", "mn2sonn"], required=True)
p.add_argument("--model", choices=["pointnet", "dgcnn"], required=True)
p.add_argument("--mn40"); p.add_argument("--sonn")
p.add_argument("--variant", default="PB_T50_RS")
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--npoints", type=int, default=1024)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--out", default="results")
p.add_argument("--max_train", type=int, default=0, help="limita muestras (solo pruebas rápidas)")
p.add_argument("--ckpt", default="", help="ruta de un .pt para guardar y retomar el avance")
p.add_argument("--ckpt_every", type=int, default=1)
p.add_argument("--amp", type=int, default=0, help="1 = precisión mixta fp16 (solo en GPU)")
p.add_argument("--stop_after_min", type=float, default=0, help="0 = sin límite de tiempo")
p.add_argument("--workers", type=int, default=0 if os.name == "nt" else 2, help="procesos del DataLoader; en Windows 0, porque ahí usan spawn y reejecutan este script")
a = p.parse_args()

import random
random.seed(a.seed); torch.manual_seed(a.seed); np.random.seed(a.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = bool(a.amp) and dev == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
TAG = f"{a.task}_{a.model}_s{a.seed}"
os.makedirs(a.out, exist_ok=True)

# ---------- datos ----------
if a.task == "mn40":
    Xtr, ytr, names = D.modelnet40(a.mn40, "train"); Xte, yte, _ = D.modelnet40(a.mn40, "test")
    extra = {}
elif a.task == "sonn":
    Xtr, ytr, names = D.scanobjectnn(a.sonn, "train", a.variant); Xte, yte, _ = D.scanobjectnn(a.sonn, "test", a.variant)
    extra = {}
else:  # entrena en ModelNet40 (10 clases compartidas), evalua en ambos dominios
    Xm, ym, nm = D.modelnet40(a.mn40, "train"); Xtr, ytr = D.restrict_shared(Xm, ym, nm, "mn40")
    Xm2, ym2, _ = D.modelnet40(a.mn40, "test"); Xte, yte = D.restrict_shared(Xm2, ym2, nm, "mn40")
    Xs, ys, sn = D.scanobjectnn(a.sonn, "test", a.variant); Xs, ys = D.restrict_shared(Xs, ys, sn, "sonn")
    names = D.SHARED_NAMES
    extra = {"real": (Xs, ys)}
if a.max_train:
    idx = np.random.permutation(len(ytr))[:a.max_train]; Xtr, ytr = Xtr[idx], ytr[idx]
nc = len(names)
tl = DataLoader(D.Clouds(Xtr, ytr, a.npoints, True), a.bs, shuffle=True, drop_last=True, num_workers=a.workers)
vl = DataLoader(D.Clouds(Xte, yte, a.npoints), 64)

# ---------- modelo y optimizador ----------
net = build(a.model, nc).to(dev)
nparams = sum(q.numel() for q in net.parameters())
if a.model == "pointnet":
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, 20, 0.7)
else:
    opt = torch.optim.SGD(net.parameters(), 0.1, momentum=0.9, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=1e-3)


def evaluate(loader):
    net.eval(); P, Y = [], []
    with torch.no_grad():
        for x, y in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = net(x.to(dev))
            P.append(out.argmax(1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def metrics(P, Y):
    cm = confusion_matrix(Y, P, labels=range(nc))
    pc = np.diag(cm) / np.maximum(cm.sum(1), 1)
    return {"oa": float((P == Y).mean()), "macc": float(pc[cm.sum(1) > 0].mean()),
            "f1_macro": float(f1_score(Y, P, average="macro", labels=range(nc), zero_division=0)),
            "per_class_acc": pc.tolist(), "confusion": cm.tolist()}


# ---------- entrenamiento ----------
hist, t0, start = [], time.time(), 0
if a.ckpt and os.path.exists(a.ckpt):   # retoma si hay avance guardado
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"]); sch.load_state_dict(ck["sch"])
    if "scaler" in ck: scaler.load_state_dict(ck["scaler"])
    hist, start = ck["hist"], ck["epoch"]; t0 -= ck.get("elapsed", 0)
    print(f"Retomando desde la época {start}", flush=True)
t_session, interrumpido = time.time(), False
for ep in range(start, a.epochs):
    net.train(); tot, n = 0.0, 0
    for x, y in tl:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = net(x)
        loss = F.cross_entropy(out.float(), y, label_smoothing=0.1)
        if a.model == "pointnet": loss = loss + 0.001 * net.reg_loss().float()
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tot += loss.item() * len(y); n += len(y)
    sch.step()
    if not np.isfinite(tot):
        print(f"ERROR {TAG}: la pérdida se volvió NaN en la época {ep+1}. Prueba con --amp 0.", flush=True); sys.exit(4)
    P, Y = evaluate(vl)
    hist.append({"epoch": ep + 1, "loss": tot / n, "test_oa": float((P == Y).mean())})
    print(f"ep {ep+1:3d}/{a.epochs} loss {tot/n:.4f} test_oa {hist[-1]['test_oa']:.4f}", flush=True)
    if a.ckpt and ((ep + 1) % a.ckpt_every == 0 or ep + 1 == a.epochs):
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(), "scaler": scaler.state_dict(),
                    "hist": hist, "epoch": ep + 1, "elapsed": time.time() - t0}, a.ckpt + ".tmp")
        os.replace(a.ckpt + ".tmp", a.ckpt)   # renombrar es atómico: el archivo nunca queda a medias
    if a.stop_after_min and ep + 1 < a.epochs and (time.time() - t_session) / 60 >= a.stop_after_min:
        if a.ckpt and (ep + 1) % a.ckpt_every != 0:   # asegura que el avance quede guardado
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(), "scaler": scaler.state_dict(),
                        "hist": hist, "epoch": ep + 1, "elapsed": time.time() - t0}, a.ckpt + ".tmp"); os.replace(a.ckpt + ".tmp", a.ckpt)
        interrumpido = True; break
if interrumpido:
    print(f"PAUSA {TAG}: van {len(hist)}/{a.epochs} épocas. Avance guardado.", flush=True); sys.exit(3)
train_time = time.time() - t0

# ---------- evaluación final (modelo de la ÚLTIMA época, sin elegir la mejor sobre test) ----------
P, Y = evaluate(vl)
res = {"task": a.task, "model": a.model, "args": vars(a), "device": dev, "params_M": nparams / 1e6,
       "train_time_s": train_time, "history": hist, "class_names": names, "test": metrics(P, Y)}
t = time.time(); evaluate(vl); res["ms_per_sample"] = (time.time() - t) / len(yte) * 1000
if "real" in extra:
    Xs, ys = extra["real"]
    Ps, Ys = evaluate(DataLoader(D.Clouds(Xs, ys, a.npoints), 64))
    res["real"] = metrics(Ps, Ys); res["real"]["n"] = int(len(ys))
    np.savez(f"{a.out}/{TAG}_samples.npz", X=D.normalize(Xs[:, :, :3])[:, :a.npoints], y=Ys, p=Ps)
else:
    np.savez(f"{a.out}/{TAG}_samples.npz", X=D.normalize(Xte[:, :, :3])[:, :a.npoints], y=Y, p=P)
res["n_train"], res["n_test"] = int(len(ytr)), int(len(yte))
res["amp"] = use_amp
json.dump(res, open(f"{a.out}/{TAG}.json", "w"), indent=1)
print("OA", res["test"]["oa"], "mAcc", res["test"]["macc"], "F1", res["test"]["f1_macro"])
