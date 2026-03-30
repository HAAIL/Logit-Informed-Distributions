#!/bin/bash

# LoID Prior Extraction and Bayesian Inference
# Runs on all datasets with include=T in split_info.csv
# Organized by task type for clarity

MODEL="gemma-2"
ALPHA=0.2
BETA=2.0
NUM_SENTENCES=10
STD_METHOD="variance"

# ============================================================
# BINARY CLASSIFICATION DATASETS (8 datasets)
# ============================================================

echo "============================================================"
echo "RUNNING BINARY CLASSIFICATION DATASETS"
echo "============================================================"

python prior_prediction.py \
    --datasets adult cervical_cancer chronic_kidney_disease bank blood \
               stroke heart_failure diabetes_prediction \
    --model $MODEL \
    --alpha $ALPHA \
    --beta $BETA \
    --num_sentences $NUM_SENTENCES \
    --std_method $STD_METHOD \
    --use_ood_split \
    --ignore_previous_run

# ============================================================
# REGRESSION DATASETS (5 datasets)
# ============================================================

echo ""
echo "============================================================"
echo "RUNNING REGRESSION DATASETS"
echo "============================================================"

python prior_prediction.py \
    --datasets airfoil_noise bike_sharing combined_cycle_power \
               concrete_strength communities_crime \
    --model $MODEL \
    --alpha $ALPHA \
    --beta $BETA \
    --num_sentences $NUM_SENTENCES \
    --std_method $STD_METHOD \
    --use_ood_split \
    --ignore_previous_run

# ============================================================
# SUMMARY
# ============================================================

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""
echo "Included datasets (13 total):"
echo "  Binary Classification: 8 datasets"
echo "  Regression: 5 datasets"
echo ""
echo "Results saved to:"
echo "  - LLM_predictions/Priors/"
echo "  - LLM_predictions/Posteriors/"
echo ""
