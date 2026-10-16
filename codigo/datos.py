"""
datos.py — Fases 1 y 2 de la propuesta (Capítulo 4).

Contiene:
  * Preprocesamiento común de textos (Sección 4.3.5).
  * Carga de MERGE Lyrics Complete con sus particiones oficiales 70-15-15.
  * Traducción verso por verso con NLLB-200 (600M destilado), con caché
    para reanudar si se interrumpe.
  * Muestra estratificada para el control de calidad (25 letras por cuadrante).
  * Carga del subconjunto en español de BRIGHTER (particiones oficiales).
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

EMOCIONES = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
CUADRANTES = ["Q1", "Q2", "Q3", "Q4"]


# ---------------------------------------------------------------------------
# Preprocesamiento (igual para ambos corpus)
# ---------------------------------------------------------------------------

_MARCA_SECCION = re.compile(r"^\s*[\[\(][^\]\)]*[\]\)]\s*$")  # [Chorus], (Verse 2)


def limpiar_texto(texto: str) -> str:
    """Normaliza espacios, quita marcas de sección entre corchetes y
    colapsa líneas vacías consecutivas. NO pasa a minúsculas ni quita
    puntuación ni palabras vacías (el codificador se preentrenó sin normalizar)."""
    if not isinstance(texto, str):
        return ""
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    lineas = []
    for linea in texto.split("\n"):
        linea = re.sub(r"[ \t ]+", " ", linea).strip()
        if _MARCA_SECCION.match(linea):
            continue
        lineas.append(linea)
    salida, previa_vacia = [], True
    for linea in lineas:
        if linea == "":
            if not previa_vacia:
                salida.append("")
            previa_vacia = True
        else:
            salida.append(linea)
            previa_vacia = False
    return "\n".join(salida).strip()


# ---------------------------------------------------------------------------
# MERGE Lyrics Complete
# ---------------------------------------------------------------------------

def _leer_tabla(ruta: Path) -> pd.DataFrame:
    if ruta.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(ruta)
    df = pd.read_csv(ruta)
    if df.shape[1] == 1 and any(sep in str(df.columns[0]) for sep in (";", "\t")):
        df = pd.read_csv(ruta, sep=";" if ";" in str(df.columns[0]) else "\t")
    return df


def _columna(df: pd.DataFrame, *claves):
    """Devuelve el nombre de la primera columna que contenga alguna clave."""
    for clave in claves:
        for c in df.columns:
            if clave in str(c).lower():
                return c
    return None


def explorar_merge(raiz: str) -> None:
    """Imprime todas las tablas y la cantidad de .txt que hay dentro de la
    carpeta descomprimida. Úsalo si cargar_merge() falla, para ver los nombres
    reales de archivos y columnas."""
    raiz = Path(raiz)
    for ruta in sorted(raiz.rglob("*")):
        if ruta.suffix.lower() in (".csv", ".xlsx", ".xls"):
            try:
                df = _leer_tabla(ruta)
                print(f"{ruta.relative_to(raiz)}  ->  {df.shape}  columnas={list(df.columns)[:12]}")
            except Exception as e:  # noqa: BLE001
                print(f"{ruta.relative_to(raiz)}  (no se pudo leer: {e})")
    txts = list(raiz.rglob("*.txt"))
    print(f"\nArchivos .txt de letras encontrados: {len(txts)}")
    if txts:
        print("Ejemplos:", [t.name for t in txts[:5]])


def _normalizar_cuadrante(q) -> str:
    q = str(q).strip().upper()
    if q in ("1", "2", "3", "4"):
        q = "Q" + q
    return q


def cargar_merge(raiz: str, perfil: str = "70_15_15",
                 archivo_metadatos: str = None, archivo_av: str = None,
                 archivos_particion: dict = None) -> pd.DataFrame:
    """Carga MERGE Lyrics Complete descomprimido en `raiz`.

    Detecta automáticamente:
      * la tabla de metadatos (tiene columna de cuadrante),
      * la tabla de valores AV (columnas de arousal y valence),
      * las letras (.txt cuyo nombre es el identificador de la canción),
      * las particiones del perfil indicado (70-15-15 por defecto).

    Si la detección falla, ejecuta explorar_merge(raiz) y pasa las rutas a
    mano con los argumentos opcionales (relativas a `raiz`).

    Devuelve un DataFrame con columnas:
      id, cuadrante, valencia, activacion, letra_en, particion
    """
    raiz = Path(raiz)
    tablas = {}
    for ruta in raiz.rglob("*"):
        if ruta.suffix.lower() in (".csv", ".xlsx", ".xls") and not ruta.name.startswith("."):
            try:
                tablas[ruta] = _leer_tabla(ruta)
            except Exception:  # noqa: BLE001
                pass
    if not tablas:
        raise FileNotFoundError(f"No encontré tablas CSV/XLSX dentro de {raiz}")

    # --- valores de valencia y activación -------------------------------
    if archivo_av:
        av = _leer_tabla(raiz / archivo_av)
    else:
        candidatas = [df for df in tablas.values()
                      if _columna(df, "arousal") and _columna(df, "valence")]
        if not candidatas:
            raise ValueError("No encontré una tabla con columnas de arousal y valence. "
                             "Ejecuta explorar_merge() y usa archivo_av=...")
        av = max(candidatas, key=len)
    col_id_av = _columna(av, "song", "id")
    av = av.rename(columns={col_id_av: "id",
                            _columna(av, "arousal"): "activacion",
                            _columna(av, "valence"): "valencia"})
    av = av[["id", "valencia", "activacion"] +
            [c for c in av.columns if "quadrant" in str(c).lower()]]

    # --- metadatos con cuadrante ------------------------------------------
    if "quadrant" not in " ".join(map(str, av.columns)).lower():
        if archivo_metadatos:
            meta = _leer_tabla(raiz / archivo_metadatos)
        else:
            candidatas = [df for df in tablas.values() if _columna(df, "quadrant")]
            if not candidatas:
                raise ValueError("No encontré la columna de cuadrante. Usa archivo_metadatos=...")
            meta = max(candidatas, key=len)
        meta = meta.rename(columns={_columna(meta, "song", "id"): "id"})
        av = av.merge(meta[["id", _columna(meta, "quadrant")]], on="id", how="left")
    av = av.rename(columns={_columna(av, "quadrant"): "cuadrante"})
    av["cuadrante"] = av["cuadrante"].map(_normalizar_cuadrante)
    av["id"] = av["id"].astype(str).str.strip()

    # --- letras -----------------------------------------------------------
    letras = {}
    for t in raiz.rglob("*.txt"):
        letras[t.stem.strip()] = t.read_text(encoding="utf-8", errors="replace")
    if not letras:
        # algunas versiones guardan la letra dentro de una tabla
        col_letra = None
        for df in tablas.values():
            col_letra = _columna(df, "lyric")
            if col_letra:
                df = df.rename(columns={_columna(df, "song", "id"): "id"})
                letras = dict(zip(df["id"].astype(str), df[col_letra]))
                break
    if not letras:
        raise ValueError("No encontré las letras (.txt ni columna 'lyrics').")
    av["letra_en"] = av["id"].map(letras)

    # --- particiones ------------------------------------------------------
    av["particion"] = None
    if archivos_particion:
        for nombre, rel in archivos_particion.items():
            ids = set(_leer_tabla(raiz / rel).iloc[:, 0].astype(str).str.strip())
            av.loc[av["id"].isin(ids), "particion"] = nombre
    else:
        etiquetas = {"train": "train", "valid": "validacion", "val": "validacion",
                     "test": "test"}
        perfil_regex = re.compile(perfil.replace("_", r"[_\- ]?"))
        for ruta, df in tablas.items():
            ruta_txt = str(ruta.relative_to(raiz)).lower()
            if not perfil_regex.search(ruta_txt):
                continue
            col_id = _columna(df, "song", "id") or df.columns[0]
            # Caso 1: un archivo por partición (…train….csv)
            nombre = next((v for k, v in etiquetas.items() if k in ruta.stem.lower()), None)
            if nombre:
                ids = set(df[col_id].astype(str).str.strip())
                av.loc[av["id"].isin(ids), "particion"] = nombre
                continue
            # Caso 2: un archivo con una columna que dice train/validate/test
            for c in df.columns:
                valores = set(df[c].astype(str).str.lower().unique())
                if {"train", "test"} <= valores:
                    mapa = {k: v for k, v in etiquetas.items()}
                    for _, fila in df.iterrows():
                        v = str(fila[c]).lower()
                        destino = next((mapa[k] for k in mapa if v.startswith(k)), None)
                        av.loc[av["id"] == str(fila[col_id]).strip(), "particion"] = destino
                    break

    faltan_letra = av["letra_en"].isna().sum()
    faltan_part = av["particion"].isna().sum()
    if faltan_letra:
        print(f"AVISO: {faltan_letra} canciones sin letra; se descartan.")
    if faltan_part:
        print(f"AVISO: {faltan_part} canciones sin partición {perfil}; se descartan. "
              "Si son muchas, revisa con explorar_merge() y pasa archivos_particion=...")
    av = av.dropna(subset=["letra_en", "particion"]).reset_index(drop=True)
    av["letra_en"] = av["letra_en"].map(limpiar_texto)

    vmin, vmax = av[["valencia", "activacion"]].min().min(), av[["valencia", "activacion"]].max().max()
    if vmin >= 0 and vmax <= 1:
        av[["valencia", "activacion"]] = 2 * av[["valencia", "activacion"]] - 1
        print("V y A estaban en [0, 1]; se reescalaron a [-1, 1] con 2x - 1.")
    return av[["id", "cuadrante", "valencia", "activacion", "letra_en", "particion"]]


def resumen_merge(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla cuadrante x partición, para verificar contra el artículo."""
    return pd.crosstab(df["cuadrante"], df["particion"], margins=True)


# ---------------------------------------------------------------------------
# Traducción con NLLB-200
# ---------------------------------------------------------------------------

def versos_unicos(letras) -> list:
    """Versos distintos (no vacíos) de todo el corpus. Los repetidos, como los
    estribillos, aparecen una sola vez y se traducen una sola vez."""
    vistos = set()
    for letra in letras:
        for verso in letra.split("\n"):
            v = verso.strip()
            if v:
                vistos.add(v)
    return sorted(vistos, key=len)  # ordenar por longitud acelera los lotes


def traducir_versos(versos, ruta_cache: str,
                    modelo_id: str = "facebook/nllb-200-distilled-600M",
                    tam_lote: int = 64, guardar_cada: int = 20,
                    max_tokens: int = 200, num_beams: int = 4) -> dict:
    """Traduce inglés -> español verso por verso, con caché JSON en `ruta_cache`.
    Si se corta la sesión, vuelve a llamar la función y continúa donde quedó.
    La decodificación es determinista (beam search, sin muestreo)."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    cache = {}
    if os.path.exists(ruta_cache):
        with open(ruta_cache, encoding="utf-8") as f:
            cache = json.load(f)
    pendientes = [v for v in versos if v not in cache]
    print(f"Versos únicos: {len(versos)} | ya traducidos: {len(versos) - len(pendientes)} "
          f"| pendientes: {len(pendientes)}")
    if not pendientes:
        return cache

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(modelo_id, src_lang="eng_Latn")
    modelo = AutoModelForSeq2SeqLM.from_pretrained(
        modelo_id, torch_dtype=torch.float16 if dispositivo == "cuda" else torch.float32
    ).to(dispositivo).eval()
    id_espanol = tok.convert_tokens_to_ids("spa_Latn")

    def guardar():
        tmp = ruta_cache + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        os.replace(tmp, ruta_cache)

    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        tqdm = lambda x, **k: x  # noqa: E731

    lotes = [pendientes[i:i + tam_lote] for i in range(0, len(pendientes), tam_lote)]
    for n, lote in enumerate(tqdm(lotes, desc="Traduciendo")):
        enc = tok(lote, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_tokens).to(dispositivo)
        with torch.no_grad():
            salida = modelo.generate(**enc, forced_bos_token_id=id_espanol,
                                     num_beams=num_beams, do_sample=False,
                                     max_new_tokens=int(enc["input_ids"].shape[1] * 1.6) + 10)
        for orig, trad in zip(lote, tok.batch_decode(salida, skip_special_tokens=True)):
            cache[orig] = trad.strip()
        if (n + 1) % guardar_cada == 0:
            guardar()
    guardar()
    return cache


def aplicar_traduccion(letra: str, cache: dict) -> str:
    """Reconstruye la letra en español conservando la segmentación en versos."""
    return "\n".join(cache.get(v.strip(), v) if v.strip() else "" for v in letra.split("\n"))


def detectar_problemas_traduccion(original: str, traduccion: str) -> list:
    """Señales automáticas de 'errores graves' (Sección 4.3.3) para priorizar la
    revisión manual. No sustituyen la revisión humana."""
    problemas = []
    for o, t in zip(original.split("\n"), traduccion.split("\n")):
        o, t = o.strip(), t.strip()
        if not o:
            continue
        if not t:
            problemas.append("verso omitido")
        elif o.lower() == t.lower() and len(o.split()) > 3:
            problemas.append("verso sin traducir")
        elif len(t.split()) > 3 * max(len(o.split()), 1) + 5:
            problemas.append("repetición anómala")
    return sorted(set(problemas))


def muestra_control_calidad(df: pd.DataFrame, n_por_cuadrante: int = 25,
                            semilla: int = 42) -> pd.DataFrame:
    """Muestra aleatoria estratificada (25 por cuadrante = 100) con columnas
    vacías para que califiques a mano (Sección 4.3.3)."""
    muestra = df.groupby("cuadrante").sample(
        n=min(n_por_cuadrante, df["cuadrante"].value_counts().min()), random_state=semilla)
    muestra = muestra[["id", "cuadrante", "letra_en", "letra_es"]].copy()
    muestra["alertas_automaticas"] = [
        "; ".join(detectar_problemas_traduccion(o, t))
        for o, t in zip(muestra["letra_en"], muestra["letra_es"])]
    muestra["fidelidad_semantica (si/parcial/no)"] = ""
    muestra["tono_conservado (si/no)"] = ""
    muestra["error_grave (ninguno/sin_traducir/repeticion/omision)"] = ""
    muestra["accion (mantener/corregir/excluir)"] = ""
    muestra["notas"] = ""
    return muestra.reset_index(drop=True)


# ---------------------------------------------------------------------------
# BRIGHTER (español)
# ---------------------------------------------------------------------------

def cargar_brighter(repo: str = "brighter-dataset/BRIGHTER-emotion-categories",
                    idioma: str = "esp") -> dict:
    """Devuelve {'train': df, 'validacion': df, 'test': df} con columnas
    id, texto y las seis emociones (0/1)."""
    from datasets import get_dataset_config_names, load_dataset

    configs = get_dataset_config_names(repo)
    if idioma not in configs:
        raise ValueError(f"'{idioma}' no está entre las configuraciones: {configs}")
    ds = load_dataset(repo, idioma)
    nombres = {"train": "train", "dev": "validacion", "validation": "validacion", "test": "test"}
    salida = {}
    for split in ds:
        df = ds[split].to_pandas()
        df = df.rename(columns={"text": "texto"})
        df["texto"] = df["texto"].map(limpiar_texto)
        for e in EMOCIONES:
            df[e] = df[e].fillna(0).astype(int)
        repetidos = df.duplicated(subset="texto", keep=False)
        if repetidos.any():
            distintas = df[repetidos].groupby("texto")[EMOCIONES].nunique().gt(1).any(axis=1).sum()
            antes = len(df)
            df = df.drop_duplicates(subset="texto", keep="first").reset_index(drop=True)
            print(f"{split}: {antes - len(df)} filas duplicadas eliminadas ({antes} -> {len(df)}); "
                  f"textos repetidos con etiquetas distintas: {distintas}")
        salida[nombres.get(split, split)] = df[["id", "texto"] + EMOCIONES]
    return salida


def resumen_brighter(particiones: dict) -> pd.DataFrame:
    filas = {n: {**df[EMOCIONES].sum().to_dict(), "instancias": len(df)}
             for n, df in particiones.items()}
    return pd.DataFrame(filas).T[["instancias"] + EMOCIONES]


# ---------------------------------------------------------------------------
# Longitud en tokens (para reportar cuántas letras superan 512)
# ---------------------------------------------------------------------------

def proporcion_truncadas(textos, tokenizer, max_len: int = 512) -> dict:
    largos = np.array([len(tokenizer(t, add_special_tokens=True)["input_ids"]) for t in textos])
    return {"media_tokens": float(largos.mean()),
            "mediana_tokens": float(np.median(largos)),
            "max_tokens": int(largos.max()),
            "proporcion_mayor_512": float((largos > max_len).mean())}
