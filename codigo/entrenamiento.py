"""
entrenamiento.py — Procedimiento de entrenamiento (Sección 4.5, Algoritmo 1).

Un solo bucle sirve para ambas configuraciones:
  * Tarea única (TU): p = 1.0, solo lotes de MERGE.
  * Multitarea  (MT): p = 0.7, lotes alternados MERGE / BRIGHTER.
En esta primera entrega (30 %) se ejecuta la tarea única; la multitarea queda
lista para la siguiente fase cambiando p y pasando los datos de BRIGHTER.
"""

import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from datos import EMOCIONES
from metricas import metricas_cuadrante, metricas_regresion

HIPERPARAMETROS = {          # Cuadro 4.3
    "lr": 2e-5,
    "weight_decay": 0.01,
    "warmup": 0.10,
    "batch_size": 16,
    "epocas": 4,
    "max_len": 512,
    "dropout": 0.1,
    "p": 1.0,                # 0.7 en multitarea
    "micro_lote": None,      # p. ej. 4 en una GPU de 6 GB
}


def fijar_semilla(semilla: int):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)


class DatosTexto(Dataset):
    def __init__(self, textos, etiquetas):
        self.textos = list(textos)
        self.etiquetas = np.asarray(etiquetas, dtype=np.float32)

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, i):
        return self.textos[i], self.etiquetas[i]


def _colate(tokenizer, max_len):
    def f(lote):
        textos, y = zip(*lote)
        enc = tokenizer(list(textos), padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"], torch.tensor(np.stack(y))
    return f


def cargador(df, columna_texto, columnas_y, tokenizer, max_len, batch_size,
             barajar, generador=None):
    ds = DatosTexto(df[columna_texto], df[columnas_y].values)
    return DataLoader(ds, batch_size=batch_size, shuffle=barajar,
                      collate_fn=_colate(tokenizer, max_len), generator=generador)


def _ciclo_infinito(loader):
    """Cuando un corpus agota sus lotes, se vuelve a barajar y sigue."""
    while True:
        for lote in loader:
            yield lote


@torch.no_grad()
def predecir_regresion(modelo, loader, dispositivo):
    modelo.eval()
    preds = []
    for ids, mask, _ in loader:
        with torch.autocast(device_type=dispositivo, dtype=torch.float16,
                            enabled=dispositivo == "cuda"):
            out = modelo(ids.to(dispositivo), mask.to(dispositivo), tarea="regresion")
        preds.append(out.float().cpu().numpy())
    return np.concatenate(preds)          # columnas: valencia, activacion


@torch.no_grad()
def predecir_clasificacion(modelo, loader, dispositivo):
    modelo.eval()
    probs = []
    for ids, mask, _ in loader:
        with torch.autocast(device_type=dispositivo, dtype=torch.float16,
                            enabled=dispositivo == "cuda"):
            out = modelo(ids.to(dispositivo), mask.to(dispositivo), tarea="clasificacion")
        probs.append(torch.sigmoid(out.float()).cpu().numpy())
    return np.concatenate(probs)


def entrenar(modelo, tokenizer, merge: pd.DataFrame, semilla: int, carpeta_salida: str,
             brighter: dict = None, hp: dict = None, columna_letra: str = "letra_es"):
    """Entrena una configuración y devuelve (resultados_test, historial).

    merge: DataFrame con columnas [columna_letra, valencia, activacion, cuadrante, particion]
    brighter: dict {'train': df, ...} con columnas texto + EMOCIONES (solo si p < 1)
    Guarda en carpeta_salida: mejor_modelo.pt, historial.json, predicciones_test.csv,
    resultados_test.json
    """
    hp = {**HIPERPARAMETROS, **(hp or {})}
    os.makedirs(carpeta_salida, exist_ok=True)
    fijar_semilla(semilla)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo.to(dispositivo)

    tr = merge[merge.particion == "train"]
    va = merge[merge.particion == "validacion"]
    te = merge[merge.particion == "test"]
    g = torch.Generator().manual_seed(semilla)
    Y = ["valencia", "activacion"]
    L_train = cargador(tr, columna_letra, Y, tokenizer, hp["max_len"], hp["batch_size"], True, g)
    L_val = cargador(va, columna_letra, Y, tokenizer, hp["max_len"], 32, False)
    L_test = cargador(te, columna_letra, Y, tokenizer, hp["max_len"], 32, False)
    it_L = _ciclo_infinito(L_train)

    p = hp["p"]
    it_A = None
    if p < 1.0:
        if brighter is None:
            raise ValueError("Con p < 1 hay que pasar brighter=...")
        A_train = cargador(brighter["train"], "texto", EMOCIONES, tokenizer,
                           hp["max_len"], hp["batch_size"], True, g)
        it_A = _ciclo_infinito(A_train)

    pasos_epoca = math.ceil(len(tr) / (p * hp["batch_size"]))      # T del Algoritmo 1
    pasos_totales = pasos_epoca * hp["epocas"]

    from transformers import get_linear_schedule_with_warmup
    sin_decaimiento = ["bias", "LayerNorm.weight", "layer_norm"]
    grupos = [
        {"params": [w for n, w in modelo.named_parameters() if w.requires_grad
                    and not any(s in n for s in sin_decaimiento)], "weight_decay": hp["weight_decay"]},
        {"params": [w for n, w in modelo.named_parameters() if w.requires_grad
                    and any(s in n for s in sin_decaimiento)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(grupos, lr=hp["lr"])
    sched = get_linear_schedule_with_warmup(opt, int(hp["warmup"] * pasos_totales), pasos_totales)
    escalador = torch.amp.GradScaler(enabled=dispositivo == "cuda")
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
    rng = random.Random(semilla)                                    # elecciones u ~ U(0,1)

    historial, mejor_mse = [], float("inf")
    ruta_mejor = os.path.join(carpeta_salida, "mejor_modelo.pt")
    for epoca in range(1, hp["epocas"] + 1):
        modelo.train()
        t0, perdidas = time.time(), {"regresion": [], "clasificacion": []}
        for _ in range(pasos_epoca):
            if it_A is None or rng.random() < p:
                tarea, (ids, mask, y) = "regresion", next(it_L)
            else:
                tarea, (ids, mask, y) = "clasificacion", next(it_A)
            ids, mask, y = ids.to(dispositivo), mask.to(dispositivo), y.to(dispositivo)
            opt.zero_grad(set_to_none=True)
            micro = hp.get("micro_lote") or len(ids)
            perdida_total = 0.0
            for i in range(0, len(ids), micro):
                sl = slice(i, i + micro)
                with torch.autocast(device_type=dispositivo, dtype=torch.float16,
                                    enabled=dispositivo == "cuda"):
                    out = modelo(ids[sl], mask[sl], tarea=tarea)
                out = out.float()
                perdida = mse(out, y[sl]) if tarea == "regresion" else bce(out, y[sl])
                perdida = perdida * (len(ids[sl]) / len(ids))
                escalador.scale(perdida).backward()
                perdida_total += perdida.item()
            escalador.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            escalador.step(opt)
            escalador.update()
            sched.step()
            perdidas[tarea].append(perdida_total)

        pv = predecir_regresion(modelo, L_val, dispositivo)
        m_val = metricas_regresion(va.valencia, va.activacion, pv[:, 0], pv[:, 1])
        fila = {"epoca": epoca, "segundos": round(time.time() - t0, 1),
                "perdida_reg": float(np.mean(perdidas["regresion"])) if perdidas["regresion"] else None,
                "perdida_cls": float(np.mean(perdidas["clasificacion"])) if perdidas["clasificacion"] else None,
                "lotes_cls": len(perdidas["clasificacion"]),
                **{f"val_{k}": float(v) for k, v in m_val.items()}}
        historial.append(fila)
        print(f"Época {epoca}: pérdida reg={fila['perdida_reg']:.4f} | "
              f"val MSE={m_val['mse_promedio']:.4f} | val R2 V={m_val['r2_valencia']:.3f} "
              f"A={m_val['r2_activacion']:.3f} | {fila['segundos']}s")
        if m_val["mse_promedio"] < mejor_mse:                       # selección por MSE val
            mejor_mse = m_val["mse_promedio"]
            torch.save(modelo.state_dict(), ruta_mejor)
            print("   ↳ mejor modelo hasta ahora, guardado")

    # Evaluación final en test con el mejor estado (el test no se toca antes)
    modelo.load_state_dict(torch.load(ruta_mejor, map_location=dispositivo))
    pt = predecir_regresion(modelo, L_test, dispositivo)
    res = metricas_regresion(te.valencia, te.activacion, pt[:, 0], pt[:, 1])
    res_q = metricas_cuadrante(te.cuadrante, pt[:, 0], pt[:, 1])
    resultados = {**{k: float(v) for k, v in res.items()},
                  "exactitud_cuadrante": float(res_q["exactitud_cuadrante"]),
                  "f1_macro_cuadrante": float(res_q["f1_macro_cuadrante"]),
                  "matriz_confusion": res_q["matriz_confusion"].tolist(),
                  "semilla": semilla, "mejor_val_mse": mejor_mse, "hiperparametros": hp}

    pd.DataFrame({"id": te["id"].values, "cuadrante": te.cuadrante.values,
                  "valencia": te.valencia.values, "activacion": te.activacion.values,
                  "valencia_pred": pt[:, 0], "activacion_pred": pt[:, 1]}
                 ).to_csv(os.path.join(carpeta_salida, "predicciones_test.csv"), index=False)
    with open(os.path.join(carpeta_salida, "historial.json"), "w") as f:
        json.dump(historial, f, indent=2)
    with open(os.path.join(carpeta_salida, "resultados_test.json"), "w") as f:
        json.dump(resultados, f, indent=2)
    return resultados, historial
