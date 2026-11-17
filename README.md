# Clasificación de nubes de puntos 3D: PointNet y DGCNN en datos sintéticos y reales

Proyecto final de la Electiva de Ingeniería de Software (Universidad de San Buenaventura, Cali).
Autor: Julian Andres Vargas Sepulveda.

Se entrenan y comparan dos arquitecturas de clasificación de nubes de puntos, **PointNet** y **DGCNN**, con el mismo preprocesamiento, el mismo aumento de datos y el mismo protocolo, para medir cuánto se pierde al pasar de objetos sintéticos (ModelNet40) a escaneos reales (ScanObjectNN).

| Experimento | Entrenamiento | Evaluación |
|---|---|---|
| **E1** | ModelNet40 (40 clases) | ModelNet40 |
| **E2** | ScanObjectNN PB_T50_RS (15 clases) | ScanObjectNN |
| **E3** | ModelNet40 (10 clases compartidas) | ModelNet40 y ScanObjectNN |

Son 18 entrenamientos en total: 2 modelos × 3 experimentos × 3 semillas, con 100 épocas cada uno.

## Resultados

Media ± desviación estándar de 3 semillas. OA: exactitud global. mAcc: exactitud media por clase.

| Experimento | Modelo | OA (%) | mAcc (%) | F1 macro (%) |
|---|---|---|---|---|
| E1, ModelNet40 | PointNet | 89,6 ± 0,6 | 85,0 ± 0,9 | 84,8 ± 0,7 |
| E1, ModelNet40 | DGCNN | 91,3 ± 0,4 | 88,4 ± 0,4 | 88,3 ± 0,3 |
| E2, ScanObjectNN | PointNet | 59,0 ± 0,9 | 52,8 ± 1,1 | 52,8 ± 1,2 |
| E2, ScanObjectNN | DGCNN | 74,6 ± 0,7 | 70,4 ± 0,7 | 71,4 ± 0,7 |

En E3 (entrenar en ModelNet40, evaluar en ScanObjectNN):

| Modelo | OA ModelNet40 (%) | OA ScanObjectNN (%) | mAcc ScanObjectNN (%) | Brecha (pp) |
|---|---|---|---|---|
| PointNet | 96,4 ± 0,6 | 33,2 ± 0,4 | 30,5 ± 0,1 | 63,2 ± 0,2 |
| DGCNN | 96,9 ± 0,4 | 36,0 ± 0,9 | 34,0 ± 0,5 | 60,8 ± 0,8 |

El cuaderno `Resultados.ipynb` recalcula estas tablas desde los 18 archivos de `results/` y comprueba que coincidan con las del informe.

## Contenido del repositorio

| Archivo | Para qué sirve |
|---|---|
| `Resultados.ipynb` | Cuaderno con los resultados ya ejecutados: tablas, comprobación contra el informe y figuras. No necesita GPU. |
| `Entrenar_Electiva_Completo_PC.ipynb` | Cuaderno para repetir los 18 entrenamientos (necesita GPU NVIDIA). |
| `run_local.py` | Lanza los 18 entrenamientos en orden y retoma donde quedó si se corta. |
| `train.py` | Entrena y evalúa un modelo en una tarea con una semilla. |
| `models.py` | PointNet (con T-Nets) y DGCNN (4 capas EdgeConv, k = 20). |
| `data.py` | Carga de datos, normalización, aumento y las 10 clases compartidas. |
| `aggregate.py` | Junta las semillas (media y desviación estándar) y genera las figuras. |
| `convertir_modelnet.py` | Convierte los archivos OFF originales de ModelNet40 a nubes de 2048 puntos en HDF5. |
| `exportar_modelos.py` | Extrae los pesos finales de los 18 modelos entrenados a `modelos/` (unos 190 MB, no incluidos en el repositorio por su tamaño). |
| `Demo_modelo.ipynb` | Carga un modelo guardado, muestra su arquitectura, comprueba que reproduce la exactitud del informe y clasifica nubes de puntos. Necesita PyTorch y los datos. |
| `results/` | Un JSON por corrida (`{tarea}_{modelo}_s{semilla}.json`) con métricas, historial y matriz de confusión. |
| `summary.json`, `summary.csv` | Resumen con media y desviación estándar. |
| `figs/` | Curvas, matrices de confusión, exactitud por clase y ejemplos de predicciones. |

## Cómo reproducir

1. Instalar PyTorch con CUDA desde <https://pytorch.org/get-started/locally/> y luego:

   ```
   pip install -r requirements.txt
   ```

2. **ModelNet40.** Se usan los modelos CAD originales de Princeton (archivos OFF). Descargar `ModelNet40.zip` del sitio oficial, descomprimir y convertir:

   ```
   python convertir_modelnet.py --entrada RUTA/ModelNet40 --salida datos/modelnet40_original_2048
   ```

   La conversión muestrea 2048 puntos sobre la superficie de cada malla (triángulos elegidos con probabilidad proporcional a su área) e intercambia los ejes y, z para que ModelNet40 y ScanObjectNN queden en el mismo sistema de coordenadas.

3. **ScanObjectNN.** Variante `PB_T50_RS` de los autores. Colocar los archivos en `datos/h5_files/main_split/`:
   `training_objectdataset_augmentedrot_scale75.h5` y `test_objectdataset_augmentedrot_scale75.h5`.

4. Entrenar (si se corta, volver a ejecutar el mismo comando):

   ```
   python run_local.py --mn40 datos/modelnet40_original_2048 --sonn datos/h5_files
   ```

   Al terminar se ejecuta `aggregate.py` y se actualizan `summary.json`, `summary.csv` y `figs/`.

Si DGCNN da falta de memoria: `--bs_dgcnn 16` (anotar el cambio). Si su pérdida da NaN: `--amp_dgcnn 0`.

## Protocolo

- 1024 puntos por nube, centrados y normalizados a una esfera unitaria.
- Aumento solo en entrenamiento: submuestreo aleatorio de puntos, escala aleatoria por eje (entre 2/3 y 3/2), traslación (±0,2) y ruido gaussiano recortado.
- Entropía cruzada con label smoothing de 0,1 y decaimiento de pesos de 1e-4 en ambos modelos.
- PointNet: Adam (lr 1e-3), StepLR (cada 20 épocas, ×0,7), regularización de la T-Net con peso 0,001. Sin precisión mixta.
- DGCNN: SGD (lr 0,1, momentum 0,9), coseno hasta 1e-3, precisión mixta fp16 con los vecinos (kNN) calculados en fp32.
- Lote de 32, 100 épocas, semillas 1, 2 y 3.
- Se reporta el modelo de la **última época**, sin elegir la mejor sobre el conjunto de prueba.

## Qué no está incluido

Por su tamaño no se suben los datasets (`datos/`), los puntos de control de los modelos (`results/avance/*.pt`, 14 a 42 MB cada uno) ni los archivos `*_samples.npz` (24 a 35 MB cada uno). Se pueden regenerar con los pasos de arriba.
