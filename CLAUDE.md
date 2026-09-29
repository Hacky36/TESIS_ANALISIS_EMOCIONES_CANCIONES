# CLAUDE.md

Contexto de la tesis para cualquier sesión de Claude Code que trabaje en este repositorio.

## Qué es este proyecto

Tesis de Ciencias de la Computación (UNSA): detección de emociones en letras de canciones en español
mediante aprendizaje multitarea (MTL) con mRoBERTa.

Pregunta de investigación: ¿en qué medida el entrenamiento multitarea con texto emocional en español,
anotado por hablantes nativos, mejora la predicción emocional de letras mediante fine-tuning de mRoBERTa?

## Diseño del modelo

- **Un solo modelo:** mRoBERTa de MarIA (`PlanTL-GOB-ES/roberta-base-bne`), codificador compartido
  por ambas tareas. BRIGHTER y MERGE son datasets, no modelos.
- **Pooling** entre el codificador y los cabezales, compartido por ambas tareas.
- **Dos cabezales** (capas pequeñas, no modelos):
  - Regresión (tarea principal, MERGE): 2 salidas, **valencia y activación**. Pérdida `MSELoss`.
  - Clasificación (tarea auxiliar, BRIGHTER): 6 salidas, emociones de Ekman (alegría, tristeza, ira,
    miedo, sorpresa, asco). **Multi-etiqueta**: usar `BCEWithLogitsLoss`, nunca `CrossEntropyLoss`.
- **Tokenización:** tokenizer de mRoBERTa, máximo 512 tokens con truncamiento.

## Datos

Solo hay dos datasets. Los datos crudos van en `data/raw/` y no se suben a git.

### MERGE Lyrics Complete (tarea principal)

- Louro et al. (2026), *MERGE: A Bimodal Audio-Lyrics Dataset for Static Music Emotion Recognition*,
  IEEE Transactions on Affective Computing, DOI 10.1109/TAFFC.2026.3668037.
- 2568 letras en inglés: Q1=600, Q2=710, Q3=621, Q4=637.
- Valores continuos de activación y valencia en [-1, 1] (archivo `av_values`), cuadrantes de Russell
  y metadatos (id, título, artista, año, género, etiquetas emocionales de AllMusic).
- Usar los splits oficiales 70-15-15 de `tvt_dataframes`, no inventar splits propios.
- Las letras se **traducen al español** con traducción automática. Es una limitación asumida y documentada.
- Advertencia: V y A **no tienen validación humana**; salen de mapear las etiquetas de AllMusic con el
  léxico de Warriner. Solo el cuadrante fue validado manualmente.

### BRIGHTER, subconjunto en español (tarea auxiliar)

- Muhammad et al. (2025), SemEval-2025 Task 11 (arXiv:2503.07269) y el paper del dataset BRIGHTER
  (arXiv:2502.11926).
- Textos cortos (no letras) en español latinoamericano, anotados por hablantes nativos, 6 emociones,
  multi-etiqueta. Guardar también el acuerdo entre anotadores que reportan: es el techo para interpretar resultados.

## Entrenamiento

- Muestreo alternante (patrón MT-DNN, Liu et al. 2019): en cada paso se elige de qué dataset viene el
  batch, con proporción de partida 70% MERGE / 30% BRIGHTER.
- Nunca mezclar ejemplos de ambos datasets en un mismo batch. Batch de MERGE activa solo la pérdida de
  regresión; batch de BRIGHTER, solo la de clasificación. Por eso no se usa ponderación de pérdidas tipo Kendall.
- Hiperparámetros de partida: batch 16, learning rate 2e-5, 4 épocas, AdamW.
- El entrenamiento completo necesita GPU y se corre en Google Colab. El código debe poder probarse
  en CPU con un subconjunto pequeño de datos.

## Ablación de pooling

- **A, token `<s>` (CLS):** falló en la versión anterior (R de valencia = -0.57). Se mantiene como línea base.
- **B, mean pooling** respetando la attention mask: resultado anterior de 0.7179 de exactitud en cuadrantes.
- **C, pooling ponderado entrenable** (atención aprendida, Yamato et al. 2023).

## Evaluación

- Regresión (V y A sobre MERGE): MSE, correlación de Pearson y R².
- Clasificación (BRIGHTER): exactitud, F1 macro y F1 micro.
- Comparaciones:
  1. MTL contra un modelo de tarea única entrenado solo con MERGE (el experimento de control que responde la pregunta).
  2. Contra la línea base publicada de MERGE (solo letras): F1 71.40% en cuadrantes, R² 0.366 en activación
     y 0.569 en valencia. Para comparar el F1, el cuadrante se **deriva del signo** de V y A predichas;
     decirlo explícitamente en la metodología.
  3. Las tres estrategias de pooling entre sí.
- Análisis cualitativo sobre fragmentos con metáfora o ambigüedad contextual.

## Decisiones tomadas (no revertir sin consultar)

- No hay dimensión de dominancia (D). El diseño es solo V y A. NRC-VAD y ANEW no se usan.
- El cabezal de clasificación no predice cuadrantes: son función determinista de V y A y no aportan señal nueva.
- No cambiar de codificador (nada de BETO, XLNet u otros) por iniciativa propia.
- Fuera de alcance salvo que se pida: segmentación jerárquica por estrofas para superar los 512 tokens.

## Estructura del repositorio

- `data/raw/`: datasets descargados (ignorado por git).
- `data/processed/`: datos traducidos y preprocesados (ignorado por git).
- `src/tesis_emociones/`: código del proyecto (datos, modelo, entrenamiento, evaluación).
- `notebooks/`: exploración y notebooks para Colab.
- `configs/`: configuraciones de experimentos.
- `results/`: métricas y figuras de los experimentos.

## Convenciones

- Código y comentarios en español o inglés, pero consistentes dentro de cada archivo.
- Fijar semillas en cada experimento para que sea reproducible.
