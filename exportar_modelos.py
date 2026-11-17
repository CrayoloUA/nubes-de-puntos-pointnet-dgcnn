"""Extrae los pesos finales (época 100) de los 18 checkpoints de results/avance
y los guarda en modelos/, sin el estado del optimizador, para poder reutilizarlos.

Cada archivo modelos/{tarea}_{modelo}_s{semilla}.pt es un diccionario con:
  net          pesos de la red (state_dict)
  model        "pointnet" o "dgcnn"
  num_classes  número de clases de salida
  class_names  nombres de las clases, en el orden de salida
  task         "mn40", "sonn" o "mn2sonn"
  seed, epochs, amp, test_oa   (test_oa: exactitud de prueba reportada en results/)

Uso:  python exportar_modelos.py
"""
import glob, json, os
import torch

os.makedirs("modelos", exist_ok=True)
total = 0
for f in sorted(glob.glob("results/*_s[0-9]*.json")):
    tag = os.path.basename(f)[:-5]
    ckpt = f"results/avance/ckpt_{tag}_e100.pt"
    r = json.load(open(f))
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert ck["epoch"] == 100, (tag, ck["epoch"])
    out = {"net": ck["net"], "model": r["model"], "num_classes": len(r["class_names"]),
           "class_names": r["class_names"], "task": r["task"], "seed": r["args"]["seed"],
           "epochs": ck["epoch"], "amp": r["amp"], "test_oa": r["test"]["oa"]}
    torch.save(out, f"modelos/{tag}.pt")
    mb = os.path.getsize(f"modelos/{tag}.pt") / 1e6
    total += mb
    print(f"{tag:<22} {mb:6.1f} MB")
print(f"Total: {total:.0f} MB en {len(glob.glob('modelos/*.pt'))} archivos")
