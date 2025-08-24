#!/bin/bash

# Parallel OmniMedVQA Chest CT Scan Evaluation Script
# This script runs evaluation in parallel across multiple GPUs

# Configuration
MODEL_PATH="/home/log/Github/llava-med-v1.5-mistral-7b"
IMAGE_FOLDER="data/OmniMedVQA"
# QUESTION_FILE="data/OmniMedVQA/QA_information/Open-access/Covid-19 tianchi.json"
# QUESTION_FILE="data/OmniMedVQA/QA_information/Open-access/Covid-19 tianchi_augmented.json"
QUESTION_FILE="data/OmniMedVQA/QA_information/Open-access/Chest CT Scan_augmented.json"
# OUTPUT_FILE="eval_results/covid19_tianchi.jsonl_augmented"
OUTPUT_FILE="eval_results/chest_ct_results_crop.jsonl"
# OUTPUT_FILE="eval_results/covid19_tianchi_augmented.jsonl"
SAMPLE_RATIO=1.0

# Calculate dataset split points
echo "Calculating dataset split for parallel processing..."
TOTAL_QUESTIONS=$(python -c "
import json
with open('$QUESTION_FILE', 'r') as f:
    data = json.load(f)
sample_size = int(len(data) * $SAMPLE_RATIO)
print(sample_size)
")

HALF_SIZE=$((TOTAL_QUESTIONS / 2))
echo "Total questions to process: $TOTAL_QUESTIONS"
echo "Split: GPU 0 (0-$HALF_SIZE), GPU 1 ($HALF_SIZE-$TOTAL_QUESTIONS)"

# Create output directory
mkdir -p eval_results

# Set common environment variables
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=0
export PYTHONPATH="/home/log/Github/LLaVA-Med:$PYTHONPATH"

echo "Starting parallel evaluation..."

# Run on GPU 0 (first half)
CUDA_VISIBLE_DEVICES=0 python llava/eval/eval_pattern_matching.py \
    --model-path "$MODEL_PATH" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$QUESTION_FILE" \
    --answers-file "$OUTPUT_FILE" \
    --sample-ratio $SAMPLE_RATIO \
    --start-idx 0 \
    --end-idx $HALF_SIZE \
    --process-id "gpu0" \
    --conv-mode "vicuna_v1" \
    --temperature 0.0 &

# Run on GPU 1 (second half)  
CUDA_VISIBLE_DEVICES=1 python llava/eval/eval_pattern_matching.py \
    --model-path "$MODEL_PATH" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$QUESTION_FILE" \
    --answers-file "$OUTPUT_FILE" \
    --sample-ratio $SAMPLE_RATIO \
    --start-idx $HALF_SIZE \
    --end-idx $TOTAL_QUESTIONS \
    --process-id "gpu1" \
    --conv-mode "vicuna_v1" \
    --temperature 0.0 &

# Wait for both processes to complete
wait

echo "Merging results from both GPUs..."

# Generate file names based on OUTPUT_FILE
BASE_NAME=$(basename "$OUTPUT_FILE" .jsonl)
DIR_NAME=$(dirname "$OUTPUT_FILE")
GPU0_FILE="${DIR_NAME}/${BASE_NAME}_gpu0.jsonl"
GPU1_FILE="${DIR_NAME}/${BASE_NAME}_gpu1.jsonl"
EVAL0_FILE="${DIR_NAME}/${BASE_NAME}_evaluation_gpu0.json"
EVAL1_FILE="${DIR_NAME}/${BASE_NAME}_evaluation_gpu1.json"
MERGED_EVAL_FILE="${DIR_NAME}/${BASE_NAME}_evaluation.json"

# Merge the output files
cat "$GPU0_FILE" "$GPU1_FILE" > "$OUTPUT_FILE"

# Merge evaluation results
python -c "
import json

# Load individual evaluation results
with open('$EVAL0_FILE', 'r') as f:
    eval0 = json.load(f)
with open('$EVAL1_FILE', 'r') as f:
    eval1 = json.load(f)

# Merge results
total_correct = eval0['correct'] + eval1['correct']
total_questions = eval0['total'] + eval1['total']
merged_accuracy = total_correct / total_questions if total_questions > 0 else 0

merged_results = {
    'accuracy': merged_accuracy,
    'correct': total_correct,
    'total': total_questions,
    'gpu0_results': eval0,
    'gpu1_results': eval1
}

# Save merged evaluation
with open('$MERGED_EVAL_FILE', 'w') as f:
    json.dump(merged_results, f, indent=2)

print(f'Combined Results: {total_correct}/{total_questions} = {merged_accuracy:.3f}')
"

# Clean up individual files
rm "$GPU0_FILE" "$GPU1_FILE" "$EVAL0_FILE" "$EVAL1_FILE"

# Clean up merged results file, keep only evaluation
rm "$OUTPUT_FILE"

echo "Parallel evaluation complete! Results cleaned up."
echo "Merged evaluation saved to: $MERGED_EVAL_FILE"