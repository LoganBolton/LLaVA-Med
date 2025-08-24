import argparse
import torch
import os
import json
import random
import re
from tqdm import tqdm
import shortuuid
from pathlib import Path
import difflib

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from PIL import Image
import math
from transformers import set_seed, logging

logging.set_verbosity_error()


def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()
 
def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    most_similar_str = None
    most_similar_index = None
    highest_similarity = 0
    
    # Iterate through each string in the list
    for i, str in enumerate(str_list):
        # Calculate the similarity between the current string and the target string
        similarity = str_similarity(str, target_str)
        
        # If the current string is more similar than the previous most similar string, update the variables
        if similarity > highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity
    
    # Return the index of the most similar string
    return most_similar_index


def extract_multiple_choice_answer(response_text, options=['A', 'B', 'C', 'D']):
    """
    Extract multiple choice answer from model response using pattern matching from model_med_eval.py.
    """
    if not response_text:
        return None
    
    response_text = response_text.strip()
    
    # Pattern 1: Look for {A}, {B}, etc. (curly brackets format)
    bracket_pattern = r'\{([A-D])\}'
    match = re.search(bracket_pattern, response_text, re.IGNORECASE)
    if match:
        answer = match.group(1).upper()
        if answer in options:
            return answer
    
    # Pattern 2: Direct answer patterns
    answer_patterns = [
        r'(?:the\s+)?answer\s+is\s+([A-D])',
        r'answer\s*[:]\s*([A-D])',
        r'therefore[,\s]*(?:the\s+answer\s+is\s+)?([A-D])',
        r'corresponds\s+to\s+(?:option\s+)?([A-D])',
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            answer = match.group(1).upper()
            if answer in options:
                return answer
    
    # Pattern 3: Look for isolated letters with context
    isolated_patterns = [
        r'\b([A-D])\)',  # "A)", "B)", etc.
        r'\(([A-D])\)',  # "(A)", "(B)", etc.
        r'\b([A-D])\.',  # "A.", "B.", etc.
        r'option\s+([A-D])',  # "option A", "option B", etc.
    ]
    
    for pattern in isolated_patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        if matches:
            answer = matches[-1].upper()
            if answer in options:
                return answer
    
    return None


def load_chest_ct_questions(data_file, image_base_path, sample_ratio=0.1, start_idx=None, end_idx=None):
    """
    Load and sample questions from the Chest CT Scan dataset.
    
    Args:
        data_file (str): Path to the JSON file with questions
        image_base_path (str): Base path for images
        sample_ratio (float): Ratio of data to sample (0.1 = 10%)
        start_idx (int): Start index for dataset subset (for parallel processing)
        end_idx (int): End index for dataset subset (for parallel processing)
    
    Returns:
        list: List of sampled questions
    """
    with open(data_file, 'r') as f:
        all_questions = json.load(f)
    
    chest_ct_questions = [q for q in all_questions]
    
    random.seed(42)
    
    # Apply dataset subset if indices provided (for parallel processing)
    if start_idx is not None and end_idx is not None:
        chest_ct_questions = chest_ct_questions[start_idx:end_idx]
        sampled_questions = chest_ct_questions
    else:
        sample_size = int(len(chest_ct_questions) * sample_ratio)
        # first N questions
        sampled_questions = chest_ct_questions[:sample_size]
    
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
        
        # Add answer format instruction
        question_text += "\n\nYou may only choose ONE of the options (A, B, C, or D)."
        
        formatted_q = {
            'question_id': q['question_id'],
            'image': os.path.relpath(image_path, image_base_path),
            'text': question_text,
            'gt_answer': q['gt_answer'],
            'zoom': q.get('zoom'),  # Preserve zoom metadata
            'options': {
                'A': q.get('option_A', ''),
                'B': q.get('option_B', ''),
                'C': q.get('option_C', ''),
                'D': q.get('option_D', '')
            }
        }
        formatted_questions.append(formatted_q)
    
    return formatted_questions


def MedicalEval(pred_dict: list) -> tuple:
    """
    Evaluate medical predictions using string similarity matching (from model_med_eval.py).
    """
    tot = len(pred_dict)
    succ = 0
    for data in pred_dict:
        try:
            a, b, c, d = data.get('option_A'), data.get('option_B'), data.get('option_C'), data.get('option_D')
            answer_list = [a, b]
            if c is not None:
                answer_list.append(c)
            if d is not None:
                answer_list.append(d)
            
            most_similar_idx = find_most_similar_index(answer_list, data['model_pred'])
            if most_similar_idx is not None and answer_list[most_similar_idx] == data['gt_answer']:
                succ += 1
                data['is_correct'] = 'yes'
            else:
                data['is_correct'] = 'no'
        except Exception as e:
            # If there's an error, mark as incorrect
            data['is_correct'] = 'no'
        
    return pred_dict, succ/tot


def evaluate_accuracy(predictions_file, questions):
    """
    Calculate accuracy using the similarity-based approach from model_med_eval.py.
    Handles multiple images per question ID with zoom metadata.
    
    Args:
        predictions_file (str): Path to file with model predictions
        questions (list): List of original questions with ground truth
    
    Returns:
        dict: Evaluation results
    """
    # Load predictions and keep original data for prompts
    predictions = {}
    pred_dict = []
    
    with open(predictions_file, 'r') as f:
        for line in f:
            pred = json.loads(line.strip())
            
            # Create unique key using question_id + zoom (if available)
            question_id = pred['question_id']
            zoom = pred.get('zoom', None)  # Handle cases where zoom might not exist
            
            if zoom is not None:
                unique_key = f"{question_id}_zoom_{zoom}"
            else:
                unique_key = question_id
                
            predictions[unique_key] = pred
            
            # Find the corresponding question data using the same unique key logic
            question_data = None
            for q in questions:
                q_id = q['question_id']
                q_zoom = q.get('zoom', None)
                
                if q_zoom is not None:
                    q_unique_key = f"{q_id}_zoom_{q_zoom}"
                else:
                    q_unique_key = q_id
                    
                if q_unique_key == unique_key:
                    question_data = q
                    break
            
            if question_data:
                eval_data = {
                    'question_id': unique_key,  # Use unique key as question_id
                    'original_question_id': question_id,  # Keep original for reference
                    'zoom': zoom,  # Keep zoom metadata
                    'model_pred': pred['text'],
                    'gt_answer': question_data['gt_answer'],
                    'option_A': question_data['options']['A'],
                    'option_B': question_data['options']['B'],
                    'option_C': question_data['options']['C'],
                    'option_D': question_data['options']['D']
                }
                pred_dict.append(eval_data)
    
    # Use MedicalEval to calculate accuracy
    evaluated_pred_dict, accuracy = MedicalEval(pred_dict)
    
    # Convert to the expected return format
    detailed_results = []
    correct = 0
    total = len(evaluated_pred_dict)
    
    for data in evaluated_pred_dict:
        is_correct = data['is_correct'] == 'yes'
        if is_correct:
            correct += 1
            
        # Get the prompt from original predictions using unique key
        unique_key = data['question_id']
        prompt = predictions.get(unique_key, {}).get('prompt', '')
        
        detailed_results.append({
            'question_id': data['question_id'],  # This is now the unique key
            'original_question_id': data.get('original_question_id', data['question_id']),
            'zoom': data.get('zoom'),
            'prompt': prompt,
            'gt_answer': data['gt_answer'],
            'model_response': data['model_pred'],
            'correct': is_correct
        })
    
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
        args.sample_ratio,
        args.start_idx,
        args.end_idx
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
    if args.process_id:
        base, ext = os.path.splitext(answers_file)
        answers_file = f"{base}_{args.process_id}{ext}"
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
                "zoom": line.get("zoom"),  # Preserve zoom metadata
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
    if args.process_id:
        base, ext = os.path.splitext(eval_file)
        eval_file = f"{base}_{args.process_id}{ext}"
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
    parser.add_argument("--start-idx", type=int, default=None,
                       help="Start index for dataset subset (for parallel processing)")
    parser.add_argument("--end-idx", type=int, default=None,
                       help="End index for dataset subset (for parallel processing)")
    parser.add_argument("--process-id", type=str, default="",
                       help="Process identifier for output files (for parallel processing)")
    
    args = parser.parse_args()
    eval_model(args)