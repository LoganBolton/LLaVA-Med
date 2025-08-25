#!/bin/bash

# Parallel OmniMedVQA Chest CT Scan Evaluation Script
# This script runs evaluation in parallel across multiple GPUs
# Automatically configures virtual environment based on model type

MODEL_TYPE="medllava"  # options: "medllava", "medgemma"
MODEL_PATH="/home/log/Github/llava-med-v1.5-mistral-7b"  # Used for medllava
MODEL_NAME="google/medgemma-4b-it"  # Used for medgemma
IMAGE_FOLDER="data/OmniMedVQA"

DATA_TYPE="_contrast"
QUESTION_FILE="data/OmniMedVQA/QA_information/Open-access/Covid-19 tianchi${DATA_TYPE}.json"
OUTPUT_FILE="eval_results/${MODEL_TYPE}/covid19_tianchi${DATA_TYPE}.jsonl"
SAMPLE_RATIO=1.0

# Auto-configure environment based on model type
if [ "$MODEL_TYPE" == "medgemma" ]; then
    echo "🤖 Configuring environment for MedGemma..."
    
    # Check if we're already in the medgemma environment
    if [[ "$VIRTUAL_ENV" != *"medgemma_env"* ]]; then
        echo "🔄 Activating medgemma virtual environment..."
        if [ -d "medgemma_env" ]; then
            source medgemma_env/bin/activate
            echo "✅ MedGemma environment activated"
        else
            echo "❌ Error: medgemma_env directory not found!"
            echo "Please ensure the medgemma virtual environment is set up"
            exit 1
        fi
    else
        echo "✅ Already in MedGemma environment"
    fi
    
elif [ "$MODEL_TYPE" == "medllava" ]; then
    echo "🏥 Configuring environment for MedLLaVA..."
    
    # Deactivate any virtual environment and use base conda/system environment
    if [[ -n "$VIRTUAL_ENV" ]]; then
        echo "🔄 Deactivating virtual environment for MedLLaVA..."
        deactivate 2>/dev/null || true
        # Force reset environment variables
        unset VIRTUAL_ENV
        export PATH="/home/log/anaconda3/bin:$PATH"
        echo "✅ Switched to base environment"
    else
        echo "✅ Already in base environment"
    fi
    
else
    echo "❌ Error: Unsupported MODEL_TYPE '$MODEL_TYPE'"
    echo "Supported types: medllava, medgemma"
    exit 1
fi

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

echo "🚀 Starting parallel evaluation with $MODEL_TYPE..."

# Run on GPU 0 (first half)
if [ "$MODEL_TYPE" == "medgemma" ]; then
    CUDA_VISIBLE_DEVICES=0 python llava/eval/eval_pattern_matching.py \
        --model-type "$MODEL_TYPE" \
        --model "$MODEL_NAME" \
        --image-folder "$IMAGE_FOLDER" \
        --question-file "$QUESTION_FILE" \
        --answers-file "$OUTPUT_FILE" \
        --sample-ratio $SAMPLE_RATIO \
        --start-idx 0 \
        --end-idx $HALF_SIZE \
        --process-id "gpu0" \
        --temperature 0.0 &
elif [ "$MODEL_TYPE" == "medllava" ]; then
    CUDA_VISIBLE_DEVICES=0 python llava/eval/eval_pattern_matching.py \
        --model-type "$MODEL_TYPE" \
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
fi

# Run on GPU 1 (second half)  
if [ "$MODEL_TYPE" == "medgemma" ]; then
    CUDA_VISIBLE_DEVICES=1 python llava/eval/eval_pattern_matching.py \
        --model-type "$MODEL_TYPE" \
        --model "$MODEL_NAME" \
        --image-folder "$IMAGE_FOLDER" \
        --question-file "$QUESTION_FILE" \
        --answers-file "$OUTPUT_FILE" \
        --sample-ratio $SAMPLE_RATIO \
        --start-idx $HALF_SIZE \
        --end-idx $TOTAL_QUESTIONS \
        --process-id "gpu1" \
        --temperature 0.0 &
elif [ "$MODEL_TYPE" == "medllava" ]; then
    CUDA_VISIBLE_DEVICES=1 python llava/eval/eval_pattern_matching.py \
        --model-type "$MODEL_TYPE" \
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
fi

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

echo "📊 Model used: $MODEL_TYPE"
echo "🔧 Environment: $([ "$MODEL_TYPE" == "medgemma" ] && echo "medgemma_env virtual environment" || echo "base conda environment")"
echo "📁 Results saved to: $MERGED_EVAL_FILE"