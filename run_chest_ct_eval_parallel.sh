#!/bin/bash

# Parallel OmniMedVQA Chest CT Scan Evaluation Script
# This script runs evaluation in parallel across multiple GPUs

# Configuration
MODEL_PATH="/home/log/Github/llava-med-v1.5-mistral-7b"
IMAGE_FOLDER="data/OmniMedVQA"
QUESTION_FILE="data/OmniMedVQA/QA_information/Open-access/Chest CT Scan.json"
OUTPUT_FILE="eval_results/chest_ct_results.jsonl"
SAMPLE_RATIO=0.03

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

# Merge the output files
cat eval_results/chest_ct_results_gpu0.jsonl eval_results/chest_ct_results_gpu1.jsonl > "$OUTPUT_FILE"

# Merge evaluation results
python -c "
import json

# Load individual evaluation results
with open('eval_results/chest_ct_results_evaluation_gpu0.json', 'r') as f:
    eval0 = json.load(f)
with open('eval_results/chest_ct_results_evaluation_gpu1.json', 'r') as f:
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
with open('eval_results/chest_ct_results_evaluation.json', 'w') as f:
    json.dump(merged_results, f, indent=2)

print(f'Combined Results: {total_correct}/{total_questions} = {merged_accuracy:.3f}')
"

# Clean up individual files
rm eval_results/chest_ct_results_gpu0.jsonl
rm eval_results/chest_ct_results_gpu1.jsonl
rm eval_results/chest_ct_results_evaluation_gpu0.json
rm eval_results/chest_ct_results_evaluation_gpu1.json

echo "Parallel evaluation complete! Results merged to: $OUTPUT_FILE"
echo "Merged evaluation saved to: eval_results/chest_ct_results_evaluation.json"