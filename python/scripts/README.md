
# IA Auditor

Script que permite obtener informacion de modelos ya entrenados.
## Requisitos

```
    [Windows]
    winget install graphviz

    [Linux]
    sudo apt-get install graphviz
```

## Uso

Para iniciar el script se debe iniciar dentro del entorno virtual generado anteriormente

```
    py ia_auditor.py <MODEL_PATH> <SCALER_PATH> <ENCODER_PATH> <TEST_DATASET_FILE> (optional)<plot>
```