#!/bin/env bash
set -eux
set -o pipefail

# Customize these lists as you like
DATASETS=(airline year taxi)
METHODS=(map fmgp lla ella valla mfvi sngp)
SEEDS=(0 1 2 3 4)

BATCH_SIZE=128
OUTPUT_DIR="benchmarks/regression/results"

echo "Running BayesiPy regression benchmarks"
echo "Datasets: ${DATASETS[*]}"
echo "Methods:  ${METHODS[*]}"
echo "Seeds:    ${SEEDS[*]}"
echo

for d in "${DATASETS[@]}"; do
  for m in "${METHODS[@]}"; do
    for s in "${SEEDS[@]}"; do
      echo "-----------------------------------------------------"
      echo "Running dataset='${d}' method='${m}' seed='${s}'"
      echo "-----------------------------------------------------"
      python benchmarks/regression/regression_unified.py \
        --dataset "${d}" \
        --method "${m}" \
        --seed "${s}" \
        --config "benchmarks/regression/configs/${m}.json" \
        --output "${OUTPUT_DIR}" \
        --verbose
    done
  done
done

echo "All benchmarks finished."