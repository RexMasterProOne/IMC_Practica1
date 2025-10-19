#!/bin/bash
# ============================================================
# Script: run_experiments.sh
# Autor: ChatGPT + Rex
# Descripción:
#   Ejecuta automáticamente todas las combinaciones de arquitecturas
#   de la práctica de IMC y guarda los resultados organizados
#   por dataset y número de capas ocultas.
# ============================================================

# --- CONFIGURACIÓN GLOBAL ---
EXEC=./bin/la1             # Ruta al ejecutable
DATADIR=./dat              # Carpeta con los datasets
RESULTSDIR=./resultados    # Carpeta donde guardar los resultados
ITER=1000                  # Iteraciones
ETA=0.1                    # Tasa de aprendizaje
MU=0.9                     # Factor de momento

# --- DATASETS DISPONIBLES ---
DATASETS=("xor" "sin" "quake" "buoy_46025")

# --- ARQUITECTURAS ---
# Una capa oculta
H1=(2 4 8 32 64 100)
# Dos capas ocultas
H2=(2 4 8 32 64 100)

# --- CREAR DIRECTORIO PRINCIPAL ---
mkdir -p "$RESULTSDIR"

# --- BUCLE SOBRE CADA DATASET ---
for dataset in "${DATASETS[@]}"; do
    echo "======================================"
    echo " Ejecutando dataset: $dataset"
    echo "======================================"

    # Crear carpetas base
    DATA_RESULT_DIR="$RESULTSDIR/$dataset"
    ONE_LAYER_DIR="$DATA_RESULT_DIR/1_capa"
    TWO_LAYER_DIR="$DATA_RESULT_DIR/2_capas"
    mkdir -p "$ONE_LAYER_DIR"
    mkdir -p "$TWO_LAYER_DIR"

    # --- 1 capa oculta ---
    for h in "${H1[@]}"; do
        echo "→ Ejecutando 1 capa oculta con $h neuronas..."
        OUTFILE="$ONE_LAYER_DIR/n${h}.txt"

        # Construir argumentos en array para evitar errores con -s
        ARGS=(-t "$DATADIR/train_${dataset}.dat"
              -T "$DATADIR/test_${dataset}.dat"
              -i $ITER -l 1 -h $h -e $ETA -m $MU)

        # Agregar normalización si es el dataset buoy
        if [[ "$dataset" == "buoy_46025" ]]; then
            ARGS+=(-s)
        fi

        # Ejecutar el programa
        "$EXEC" "${ARGS[@]}" > "$OUTFILE" 2>&1
        echo "  Guardado en $OUTFILE"
    done

    # --- 2 capas ocultas ---
    for h in "${H2[@]}"; do
        echo "→ Ejecutando 2 capas ocultas con $h neuronas por capa..."
        OUTFILE="$TWO_LAYER_DIR/n${h}_${h}.txt"

        ARGS=(-t "$DATADIR/train_${dataset}.dat"
              -T "$DATADIR/test_${dataset}.dat"
              -i $ITER -l 2 -h $h -e $ETA -m $MU)

        if [[ "$dataset" == "buoy_46025" ]]; then
            ARGS+=(-s)
        fi

        "$EXEC" "${ARGS[@]}" > "$OUTFILE" 2>&1
        echo "  Guardado en $OUTFILE"
    done

    echo ""
done

echo "======================================"
echo " TODAS LAS PRUEBAS FINALIZADAS 🎉"
echo " Resultados guardados en: $RESULTSDIR"
echo "======================================"
