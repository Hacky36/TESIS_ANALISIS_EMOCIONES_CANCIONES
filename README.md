# TESIS_ANALISIS_EMOCIONES_CANCIONES

Detección de emociones en letras de canciones en español mediante aprendizaje multitarea con mRoBERTa.

- **Tarea principal:** regresión de valencia y activación sobre MERGE Lyrics Complete, traducido al español.
- **Tarea auxiliar:** clasificación multi-etiqueta de 6 emociones sobre el subconjunto en español de BRIGHTER (SemEval-2025 Task 11).
- **Modelo:** un codificador compartido `PlanTL-GOB-ES/roberta-base-bne` con dos cabezales y muestreo alternante.

El diseño completo está en [CLAUDE.md](CLAUDE.md).

## Estructura

```
data/raw/          datasets descargados (no se suben a git)
data/processed/    datos traducidos y preprocesados
src/tesis_emociones/  código del proyecto
notebooks/         exploración y notebooks para Colab
configs/           configuraciones de experimentos
results/           métricas y figuras
```

## Instalación

```bash
pip install -r requirements.txt
```
