"""
modelo.py — Configuración del modelo (Sección 4.4).

Codificador mRoBERTa compartido + agrupamiento (A, B o C) + dropout 0.1 +
dos cabezales: regresión (V, A) y clasificación multietiqueta (6 emociones).
"""

import torch
from torch import nn

# Identificador del Capítulo 4; el segundo es una copia del mismo modelo por si
# el primero dejara de estar disponible en Hugging Face.
CODIFICADORES = ["BSC-LT/mRoBERTa"]

AGRUPAMIENTOS = {"A": "inicio", "B": "media", "C": "atencion"}


def cargar_codificador(nombre: str = None, congelar_embeddings: bool = False):
    from transformers import AutoModel, AutoTokenizer
    candidatos = [nombre] if nombre else CODIFICADORES
    ultimo_error = None
    for c in candidatos:
        try:
            tok = AutoTokenizer.from_pretrained(c)
            enc = AutoModel.from_pretrained(c, add_pooling_layer=False)
            if congelar_embeddings:
                enc.embeddings.word_embeddings.weight.requires_grad_(False)
            entrenables = sum(p.numel() for p in enc.parameters() if p.requires_grad)
            total = sum(p.numel() for p in enc.parameters())
            print(f"Codificador {c}: {total/1e6:.0f} M parámetros, {entrenables/1e6:.0f} M entrenables")
            return enc, tok, c
        except Exception as e:
            print(f"No se pudo cargar {c}: {e}")
            ultimo_error = e
    raise RuntimeError("No se pudo cargar ningún codificador") from ultimo_error


class AgrupamientoAtencion(nn.Module):
    """Estrategia C: s_i = w^T h_i + b; alpha = softmax enmascarado; z = sum alpha_i h_i."""

    def __init__(self, dim: int):
        super().__init__()
        self.puntuacion = nn.Linear(dim, 1)

    def forward(self, H, mascara):
        s = self.puntuacion(H).squeeze(-1)                       # (B, n)
        s = s.masked_fill(mascara == 0, torch.finfo(s.dtype).min)
        alpha = torch.softmax(s, dim=-1)                          # (B, n)
        z = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)           # (B, d)
        return z, alpha


class ModeloEmociones(nn.Module):
    def __init__(self, codificador, agrupamiento: str = "B", dropout: float = 0.1,
                 n_emociones: int = 6):
        super().__init__()
        if agrupamiento not in AGRUPAMIENTOS:
            raise ValueError(f"agrupamiento debe ser A, B o C, no {agrupamiento}")
        self.codificador = codificador
        self.agrupamiento = agrupamiento
        dim = codificador.config.hidden_size
        self.atencion = AgrupamientoAtencion(dim) if agrupamiento == "C" else None
        self.dropout = nn.Dropout(dropout)
        self.cabezal_regresion = nn.Linear(dim, 2)              # (v, a)
        self.cabezal_clasificacion = nn.Linear(dim, n_emociones)

    def agrupar(self, H, mascara):
        if self.agrupamiento == "A":                            # token de inicio <s>
            return H[:, 0], None
        if self.agrupamiento == "B":                            # promedio sin relleno
            m = mascara.unsqueeze(-1).to(H.dtype)
            return (H * m).sum(1) / m.sum(1).clamp(min=1e-9), None
        return self.atencion(H, mascara)                        # atención

    def forward(self, input_ids, attention_mask, tarea: str = "regresion",
                devolver_pesos: bool = False):
        H = self.codificador(input_ids=input_ids,
                             attention_mask=attention_mask).last_hidden_state
        z, alpha = self.agrupar(H, attention_mask)
        z = self.dropout(z)
        if tarea == "regresion":
            salida = self.cabezal_regresion(z)                   # (B, 2): [valencia, activacion]
        elif tarea == "clasificacion":
            salida = self.cabezal_clasificacion(z)               # logits (B, 6); sigmoide fuera
        else:
            raise ValueError(tarea)
        return (salida, alpha) if devolver_pesos else salida
