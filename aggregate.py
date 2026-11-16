"""Junta todas las semillas: tablas con media y desviación estándar, y figuras.
Uso: python aggregate.py --res results --out figs
Lee results/{task}_{model}_s{seed}.json. Escribe summary.json, summary.csv y las figuras."""
import argparse, glob, json, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--res", default="results"); ap.add_argument("--out", default="figs")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
SHARED_SONN = ["bed", "chair", "desk", "door", "shelf", "display", "sink", "sofa", "table", "toilet"]
NM = {"pointnet": "PointNet", "dgcnn": "DGCNN"}
TT = {"mn40": "E1: ModelNet40", "sonn": "E2: ScanObjectNN", "mn2sonn": "E3: ModelNet40 → ScanObjectNN"}
COL = {"pointnet": "#8a8a8a", "dgcnn": "#1f4e79"}

R = {}   # (task, model) -> {seed: resultado}
for f in sorted(glob.glob(f"{a.res}/*_s[0-9]*.json")):
    m = re.match(r"(mn40|sonn|mn2sonn)_(pointnet|dgcnn)_s(\d+)\.json$", os.path.basename(f))
    if m: R.setdefault((m[1], m[2]), {})[int(m[3])] = json.load(open(f))
assert R, f"No hay resultados en {a.res}"

def ms(v):
    v = np.array(v, float); return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan"))

def shared_acc(r):   # E2 evaluado solo sobre las 10 clases compartidas (el modelo igual elige entre 15)
    cm = np.array(r["test"]["confusion"], float); idx = [r["class_names"].index(c) for c in SHARED_SONN]
    return float(np.trace(cm[np.ix_(idx, idx)]) / cm[idx].sum())

S = {}
for (t, m), runs in R.items():
    seeds = sorted(runs); rs = [runs[s] for s in seeds]; d = {"seeds": seeds, "n": len(seeds), "epochs": rs[0]["args"]["epochs"],
        "params_M": rs[0]["params_M"], "amp": rs[0].get("amp", False), "n_train": rs[0]["n_train"], "n_test": rs[0]["n_test"]}
    main = [r["real"] if t == "mn2sonn" else r["test"] for r in rs]            # E3 se reporta sobre ScanObjectNN
    for k in ("oa", "macc", "f1_macro"): d[k] = ms([x[k] for x in main]); d[k + "_por_semilla"] = [x[k] for x in main]
    if t == "mn2sonn":
        d["oa_mn40"] = ms([r["test"]["oa"] for r in rs]); d["macc_mn40"] = ms([r["test"]["macc"] for r in rs])
        d["brecha_pp"] = ms([100 * (r["test"]["oa"] - r["real"]["oa"]) for r in rs])
    if t == "sonn": d["oa_10_clases"] = ms([shared_acc(r) for r in rs])
    d["min_entrenamiento"] = ms([r["train_time_s"] / 60 for r in rs]); d["ms_por_muestra"] = ms([r["ms_per_sample"] for r in rs])
    S[f"{t}.{m}"] = d
json.dump(S, open("summary.json", "w"), indent=1, ensure_ascii=False)

with open("summary.csv", "w", encoding="utf8") as f:
    f.write("experimento,modelo,semillas,epocas,OA_media,OA_std,mAcc_media,mAcc_std,F1_media,F1_std,parametros_M,min_entrenamiento\n")
    for k, d in S.items():
        t, m = k.split(".")
        f.write(f"{t},{m},{d['n']},{d['epochs']},{100*d['oa'][0]:.2f},{100*d['oa'][1]:.2f},{100*d['macc'][0]:.2f},{100*d['macc'][1]:.2f},"
                f"{100*d['f1_macro'][0]:.2f},{100*d['f1_macro'][1]:.2f},{d['params_M']:.2f},{d['min_entrenamiento'][0]:.0f}\n")
print("Resumen (OA %, media ± desv. estándar entre semillas):")
for k, d in S.items():
    print(f"  {k:18s} n={d['n']}  OA {100*d['oa'][0]:.1f} ± {100*d['oa'][1]:.1f}   mAcc {100*d['macc'][0]:.1f}   F1 {100*d['f1_macro'][0]:.1f}"
          + (f"   brecha {d['brecha_pp'][0]:.1f} ± {d['brecha_pp'][1]:.1f} pp" if "brecha_pp" in d else ""))

# 1) curvas: media entre semillas con banda de ± 1 desviación
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
for i, t in enumerate(TT):
    for m in NM:
        runs = R.get((t, m))
        if not runs: continue
        H = np.array([[h["test_oa"] * 100 for h in r["history"]] for r in runs.values()]); x = np.arange(1, H.shape[1] + 1)
        ax[i].plot(x, H.mean(0), color=COL[m], label=f"{NM[m]} (n={len(H)})", lw=1.4)
        if len(H) > 1: ax[i].fill_between(x, H.mean(0) - H.std(0, ddof=1), H.mean(0) + H.std(0, ddof=1), color=COL[m], alpha=.18, lw=0)
    ax[i].set_title(TT[t], fontsize=8); ax[i].set_xlabel("Época"); ax[i].grid(alpha=.25)
ax[0].set_ylabel("Exactitud en prueba (%)"); ax[0].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(f"{a.out}/curvas.png", dpi=220); plt.close(fig)

# 2) matrices de confusión normalizadas por fila, sumando las semillas de cada modelo
for (t, m), runs in R.items():
    rs = list(runs.values()); key = "real" if t == "mn2sonn" else "test"
    cm = sum(np.array(r[key]["confusion"], float) for r in rs); cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    names = rs[0]["class_names"]; n = len(cm); s = 5.2 if n > 20 else 3.6
    fig, ax = plt.subplots(figsize=(s, s)); im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n)); fs = 5 if n > 20 else 7
    ax.set_xticklabels(names, rotation=90, fontsize=fs); ax.set_yticklabels(names, fontsize=fs)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Clase real"); fig.colorbar(im, fraction=.046)
    ax.set_title(f"{NM[m]}, {TT[t]} ({len(rs)} semilla{'s' if len(rs) > 1 else ''})", fontsize=7.5)
    fig.tight_layout(); fig.savefig(f"{a.out}/confusion_{t}_{m}.png", dpi=220); plt.close(fig)

# 3) exactitud por clase en las 10 clases compartidas: E3 en ModelNet40, E3 en ScanObjectNN
for m in NM:
    runs = R.get(("mn2sonn", m))
    if not runs: continue
    rs = list(runs.values()); names = rs[0]["class_names"]
    pm = np.array([r["test"]["per_class_acc"] for r in rs]) * 100; pr = np.array([r["real"]["per_class_acc"] for r in rs]) * 100
    fig, ax = plt.subplots(figsize=(7.2, 2.6)); x = np.arange(len(names)); w = .38
    ax.bar(x - w / 2, pm.mean(0), w, color="#8a8a8a", label="ModelNet40 (sintético)")
    ax.bar(x + w / 2, pr.mean(0), w, color="#1f4e79", label="ScanObjectNN (real)")
    if len(rs) > 1:
        ax.errorbar(x - w / 2, pm.mean(0), pm.std(0, ddof=1), fmt="none", ecolor="k", lw=.7, capsize=1.5)
        ax.errorbar(x + w / 2, pr.mean(0), pr.std(0, ddof=1), fmt="none", ecolor="k", lw=.7, capsize=1.5)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30); ax.set_ylabel("Exactitud (%)"); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(f"{a.out}/clases_e3_{m}.png", dpi=220); plt.close(fig)

# 4) ejemplos visuales (semilla más baja de cada corrida): 3 aciertos y 3 errores
for (t, m), runs in R.items():
    s0 = min(runs); f = f"{a.res}/{t}_{m}_s{s0}_samples.npz"
    if not os.path.exists(f): continue
    z = np.load(f); names = runs[s0]["class_names"]; X, y, pr = z["X"], z["y"], z["p"]
    rng = np.random.default_rng(3); ok, bad = np.where(y == pr)[0], np.where(y != pr)[0]
    pick = list(rng.choice(ok, min(3, len(ok)), replace=False)) + list(rng.choice(bad, min(3, len(bad)), replace=False))
    fig = plt.figure(figsize=(7.2, 2.6))
    for j, i in enumerate(pick):
        ax = fig.add_subplot(1, len(pick), j + 1, projection="3d"); P = X[i]; good = y[i] == pr[i]
        ax.scatter(P[:, 0], P[:, 2], P[:, 1], s=1.2, c="#1f4e79" if good else "#b03a2e", alpha=.8)
        ax.set_axis_off(); ax.set_box_aspect((1, 1, 1))
        ax.set_title(f"real: {names[y[i]]}\npred: {names[pr[i]]}", fontsize=6.5, color="#222" if good else "#b03a2e")
    fig.tight_layout(pad=.3); fig.savefig(f"{a.out}/ejemplos_{t}_{m}.png", dpi=220); plt.close(fig)
print("figuras en", a.out)
