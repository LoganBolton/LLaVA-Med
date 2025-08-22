import argparse
import torch
import os
import json
import random
import re
from tqdm import tqdm
import shortuuid
from pathlib import Path

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from PIL import Image
import math
from transformers import set_seed, logging

logging.set_verbosity_error()


def extract_multiple_choice_answer(response_text, options=['A', 'B', 'C', 'D']):
    """
    Extract multiple choice answer from model response using rule-based pattern matching.
    
    Args:
        response_text (str): The model's response text
        options (list): List of valid option letters
    
    Returns:
        str: Extracted answer letter or None if no clear answer found
    """
    if not response_text:
        return None
    
    # Clean the response text
    response_text = response_text.strip()
    
    # Pattern 1: Direct answer patterns like "The answer is A", "Answer: B", etc.
    answer_patterns = [
        r'(?:the\s+)?answer\s+is\s+([A-D])',
        r'answer\s*[:]\s*([A-D])',
        r'(?:option\s+)?([A-D])\s+is\s+(?:the\s+)?correct',
        r'correct\s+(?:answer\s+is\s+)?(?:option\s+)?([A-D])',
        r'therefore[,\s]*(?:the\s+answer\s+is\s+)?([A-D])',
        r'so[,\s]*(?:the\s+answer\s+is\s+)?([A-D])',
        r'thus[,\s]*(?:the\s+answer\s+is\s+)?([A-D])',
    ]
    
    # Check for direct answer patterns (case insensitive)
    for pattern in answer_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            answer = match.group(1).upper()
            if answer in options:
                return answer
    
    # Pattern 2: Look for isolated letters with context
    isolated_patterns = [
        r'\b([A-D])\)',  # "A)", "B)", etc.
        r'\(([A-D])\)',  # "(A)", "(B)", etc.
        r'\b([A-D])\.',  # "A.", "B.", etc.
        r'option\s+([A-D])',  # "option A", "option B", etc.
    ]
    
    for pattern in isolated_patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        if matches:
            # Take the last match as it's often the final answer
            answer = matches[-1].upper()
            if answer in options:
                return answer
    
    # Pattern 3: Look for single letters that appear at the end or beginning
    # Check end of response for single letter
    end_match = re.search(r'\b([A-D])\s*$', response_text, re.IGNORECASE)
    if end_match:
        answer = end_match.group(1).upper()
        if answer in options:
            return answer
    
    # Check beginning of response for single letter
    start_match = re.search(r'^([A-D])\b', response_text, re.IGNORECASE)
    if start_match:
        answer = start_match.group(1).upper()
        if answer in options:
            return answer
    
    # Pattern 4: Count occurrences of each option and pick the most frequent
    option_counts = {}
    for option in options:
        # Count how many times each option appears in various contexts
        count = len(re.findall(r'\b' + option + r'\b', response_text, re.IGNORECASE))
        option_counts[option] = count
    
    # If one option appears significantly more than others, return it
    max_count = max(option_counts.values())
    if max_count > 0:
        most_frequent = [opt for opt, count in option_counts.items() if count == max_count]
        if len(most_frequent) == 1:
            return most_frequent[0]
    
    # If no clear answer found, return None
    return None


def load_chest_ct_questions(data_file, image_base_path, sample_ratio=0.1):
    """
    Load and sample questions from the Chest CT Scan dataset.
    
    Args:
        data_file (str): Path to the JSON file with questions
        image_base_path (str): Base path for images
        sample_ratio (float): Ratio of data to sample (0.1 = 10%)
    
    Returns:
        list: List of sampled questions
    """
    with open(data_file, 'r') as f:
        all_questions = json.load(f)
    
    # Filter for Chest CT Scan questions only
    chest_ct_questions = [q for q in all_questions if q.get('dataset') == 'Chest CT Scan']
    
    # Random sampling
    random.seed(42)  # For reproducibility
    sample_size = int(len(chest_ct_questions) * sample_ratio)
    sampled_questions = random.sample(chest_ct_questions, sample_size)
    
    # Convert to the format expected by the evaluation script
    formatted_questions = []
    for q in sampled_questions:
        # Build full image path
        image_path = os.path.join(image_base_path, q['image_path'])
        
        # Format question with multiple choice options
        question_text = q['question']
        if 'option_A' in q:
            question_text += f"\nA) {q['option_A']}"
        if 'option_B' in q:
            question_text += f"\nB) {q['option_B']}"
        if 'option_C' in q:
            question_text += f"\nC) {q['option_C']}"
        if 'option_D' in q:
            question_text += f"\nD) {q['option_D']}"
        
        formatted_q = {
            'question_id': q['question_id'],
            'image': os.path.relpath(image_path, image_base_path),
            'text': question_text,
            'gt_answer': q['gt_answer'],
            'options': {
                'A': q.get('option_A', ''),
                'B': q.get('option_B', ''),
                'C': q.get('option_C', ''),
                'D': q.get('option_D', '')
            }
        }
        formatted_questions.append(formatted_q)
    
    return formatted_questions


def evaluate_accuracy(predictions_file, questions):
    """
    Calculate accuracy by comparing extracted answers with ground truth.
    
    Args:
        predictions_file (str): Path to file with model predictions
        questions (list): List of original questions with ground truth
    
    Returns:
        dict: Evaluation results
    """
    # Load predictions
    predictions = {}
    with open(predictions_file, 'r') as f:
        for line in f:
            pred = json.loads(line.strip())
            predictions[pred['question_id']] = pred
    
    # Create ground truth mapping
    gt_mapping = {q['question_id']: q['gt_answer'] for q in questions}
    
    correct = 0
    total = 0
    detailed_results = []
    
    for q_id, gt_answer in gt_mapping.items():
        if q_id in predictions:
            pred_data = predictions[q_id]
            model_response = pred_data['text']
            extracted_answer = extract_multiple_choice_answer(model_response)
            
            is_correct = extracted_answer == gt_answer
            if is_correct:
                correct += 1
            
            detailed_results.append({
                'question_id': q_id,
                'gt_answer': gt_answer,
                'extracted_answer': extracted_answer,
                'model_response': model_response,
                'correct': is_correct
            })
            
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'detailed_results': detailed_results
    }


def eval_model(args):
    """
    Evaluate LLaVA model on OmniMedVQA Chest CT Scan dataset.
    """
    set_seed(42)
    
    # Load questions and sample
    print(f"Loading and sampling {args.sample_ratio*100}% of Chest CT Scan questions...")
    questions = load_chest_ct_questions(
        args.question_file, 
        args.image_folder, 
        args.sample_ratio
    )
    print(f"Loaded {len(questions)} questions for evaluation")
    
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )
    
    # Prepare output files
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    # Generate predictions
    print("Generating model predictions...")
    with open(answers_file, "w") as ans_file:
        for line in tqdm(questions):
            idx = line["question_id"]
            image_file = line["image"]
            qs = line["text"].replace(DEFAULT_IMAGE_TOKEN, '').strip()
            cur_prompt = qs
            
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            # Load and process image
            full_image_path = os.path.join(args.image_folder, image_file)
            image = Image.open(full_image_path)
            image_tensor = process_images([image], image_processor, model.config)[0]

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            ans_id = shortuuid.uuid()
            result = {
                "question_id": idx,
                "prompt": cur_prompt,
                "text": outputs,
                "answer_id": ans_id,
                "model_id": model_name,
                "metadata": {
                    "gt_answer": line["gt_answer"],
                    "options": line["options"]
                }
            }
            ans_file.write(json.dumps(result) + "\n")
            ans_file.flush()
    
    # Evaluate accuracy
    print("Evaluating accuracy...")
    eval_results = evaluate_accuracy(answers_file, questions)
    
    # Save evaluation results
    eval_file = args.answers_file.replace('.jsonl', '_evaluation.json')
    with open(eval_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    # Print summary
    print(f"\nEvaluation Results:")
    print(f"Accuracy: {eval_results['accuracy']:.3f}")
    print(f"Correct: {eval_results['correct']}/{eval_results['total']}")
    print(f"Results saved to: {eval_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, 
                       help="Path to LLaVA model weights")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True,
                       help="Path to OmniMedVQA Images folder")
    parser.add_argument("--question-file", type=str, required=True,
                       help="Path to Chest CT Scan.json file")
    parser.add_argument("--answers-file", type=str, required=True,
                       help="Output file for model predictions")
    parser.add_argument("--sample-ratio", type=float, default=0.1,
                       help="Ratio of dataset to sample (default: 0.1 = 10%)")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    
    args = parser.parse_args()
    eval_model(args)