#!/bin/bash

# OmniMedVQA Chest CT Scan Evaluation Script
# This script evaluates LLaVA-Med model on 10% of Chest CT Scan dataset

# Configuration
MODEL_PATH="/Users/log/Github/llava-med-v1.5-mistral-7b"
IMAGE_FOLDER="data/OmniMedVQA/Images"
QUESTION_FILE="data/OmniMedVQA/QA_information/Open-access/Chest CT Scan.json"
OUTPUT_FILE="eval_results/chest_ct_10percent_results.jsonl"
SAMPLE_RATIO=0.1

# Create output directory
mkdir -p eval_results

# Run evaluation
echo "Starting OmniMedVQA Chest CT Scan evaluation..."
echo "Model: $MODEL_PATH"
echo "Sample ratio: ${SAMPLE_RATIO} (10%)"
echo "Output: $OUTPUT_FILE"

python llava/eval/eval_pattern_matching.py \
    --model-path "$MODEL_PATH" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$QUESTION_FILE" \
    --answers-file "$OUTPUT_FILE" \
    --sample-ratio $SAMPLE_RATIO \
    --conv-mode "vicuna_v1" \
    --temperature 0.2

echo "Evaluation complete! Check results in eval_results/ directory."