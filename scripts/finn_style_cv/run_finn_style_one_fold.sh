#!/usr/bin/env bash
set -euo pipefail

FOLD="${1:?Usage: bash scripts/finn_style_cv/run_finn_style_one_fold.sh <fold_id 0-4>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

TRAIN_REL="Data/splits/finn_style/Gold700_finnstyle_train_fold${FOLD}.csv"
VAL_REL="Data/splits/finn_style/Gold700_finnstyle_val_fold${FOLD}.csv"
TEST_REL="Data/splits/finn_style/Gold700_finnstyle_test.csv"

test -f "$TRAIN_REL" || { echo "Missing $TRAIN_REL"; exit 1; }
test -f "$VAL_REL" || { echo "Missing $VAL_REL"; exit 1; }
test -f "$TEST_REL" || { echo "Missing $TEST_REL"; exit 1; }

LOG_ROOT="./<path_to_logs>/logs/runs/DDPM_2D"
TMP_DIR="/tmp/cddpm_finn/fold${FOLD}"
ENCODER_LINK="${TMP_DIR}/encoder_best.ckpt"
DDPM_LINK="${TMP_DIR}/ddpm_best.ckpt"

mkdir -p "$TMP_DIR" \
         "checkpoints/finn_style/fold${FOLD}" \
         "docs/run_notes/finn_style_cv/fold${FOLD}" \
         "docs/brats_eval/finn_style_fold${FOLD}_finnstyle_t500"

echo "============================================================"
echo "FINN-STYLE CV: FOLD ${FOLD}"
echo "TRAIN_CSV=$(realpath "$TRAIN_REL")"
echo "VAL_CSV=$(realpath "$VAL_REL")"
echo "HEALTHY_TEST_CSV=$(realpath "$TEST_REL")"
echo "Started: $(date)"
echo "============================================================"

echo
echo "=== Stage 1: train Spark_3D encoder, fold ${FOLD} ==="
python run.py model=Spark_3D datamodule=Gold_700_finn_fold${FOLD} logger=csv \
  test_after_training=False \
  trainer.precision=16 \
  trainer.max_epochs=500

echo
echo "=== Find Fold ${FOLD} encoder checkpoint ==="
ENCODER_SRC=$(find "$LOG_ROOT" \
  -path "*Spark_3D_Gold_700_finn_fold${FOLD}*max_epochs-500*precision-16*/checkpoints/*.ckpt" \
  -type f ! -name "last*.ckpt" \
  -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)

echo "ENCODER_SRC=$ENCODER_SRC"
test -f "$ENCODER_SRC" || { echo "ERROR: encoder checkpoint not found"; exit 1; }

ln -sfn "$(realpath "$ENCODER_SRC")" "$ENCODER_LINK"

echo "$ENCODER_SRC" > "docs/run_notes/finn_style_cv/fold${FOLD}/encoder_best_realpath.txt"
echo "$ENCODER_LINK" > "docs/run_notes/finn_style_cv/fold${FOLD}/encoder_best_nospace_path.txt"

echo "ENCODER_LINK=$ENCODER_LINK"
ls -lh "$ENCODER_LINK"

echo
echo "=== Stage 2: train DDPM_2D_3DEnc, fold ${FOLD} ==="
python run.py model=DDPM_2D_3DEnc datamodule=Gold_700_finn_fold${FOLD} logger=csv \
  model.cfg.encoder_path="$ENCODER_LINK" \
  model.cfg.condition=true \
  test_after_training=False \
  trainer.precision=16 \
  trainer.max_epochs=1000

echo
echo "=== Find Fold ${FOLD} DDPM checkpoint ==="
DDPM_RUN_DIR=$(find "$LOG_ROOT" -maxdepth 1 -type d \
  -name "*DDPM_2D_3DEnc*Gold_700_finn_fold${FOLD}*max_epochs-1000*precision-16*" \
  -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)

echo "DDPM_RUN_DIR=$DDPM_RUN_DIR"
test -d "$DDPM_RUN_DIR" || { echo "ERROR: DDPM run dir not found"; exit 1; }

DDPM_CKPT=$(find "$DDPM_RUN_DIR/checkpoints" \
  -name "*.ckpt" -type f ! -name "last*.ckpt" \
  -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)

echo "DDPM_CKPT=$DDPM_CKPT"
test -f "$DDPM_CKPT" || { echo "ERROR: DDPM checkpoint not found"; exit 1; }

ln -sfn "$(realpath "$DDPM_CKPT")" "$DDPM_LINK"

echo "$DDPM_RUN_DIR" > "docs/run_notes/finn_style_cv/fold${FOLD}/ddpm_run_dir.txt"
echo "$DDPM_CKPT" > "docs/run_notes/finn_style_cv/fold${FOLD}/ddpm_best_realpath.txt"
echo "$DDPM_LINK" > "docs/run_notes/finn_style_cv/fold${FOLD}/ddpm_best_nospace_path.txt"
cp "$DDPM_RUN_DIR/csv/metrics.csv" "docs/run_notes/finn_style_cv/fold${FOLD}/ddpm_metrics.csv"

echo "DDPM_LINK=$DDPM_LINK"
ls -lh "$DDPM_LINK"

echo
echo "=== Stage 3: full BraTS eval, Finn-style t=500, fold ${FOLD} ==="
PYTHONPATH="$PWD:${PYTHONPATH:-}" python scripts/brats_eval/08_eval_brats_ddpm3denc_finnstyle.py \
  --csv Data/splits/Brats21_goldstyle_t2_all_absolute.csv \
  --ddpm-ckpt "$DDPM_LINK" \
  --encoder-ckpt "$ENCODER_LINK" \
  --outdir "docs/brats_eval/finn_style_fold${FOLD}_finnstyle_t500" \
  --slice-batch-size 10 \
  --save-figures 5 \
  --timesteps 500 \
  --datamodule-config "configs/datamodule/Gold_700_finn_fold${FOLD}.yaml" \
  --postprocess-finn \
  --resume

echo
echo "=== Fold ${FOLD} completed ==="
cat "docs/brats_eval/finn_style_fold${FOLD}_finnstyle_t500/brats_ddpm3denc_finnstyle_summary.csv"

echo "Finished: $(date)"
