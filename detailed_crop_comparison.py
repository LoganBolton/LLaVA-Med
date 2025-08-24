import json

# Load the original question files to get image paths
with open('/home/log/Github/LLaVA-Med/data/OmniMedVQA/QA_information/Open-access/Chest CT Scan_augmented.json', 'r') as f:
    augmented_questions = json.load(f)

with open('/home/log/Github/LLaVA-Med/data/OmniMedVQA/QA_information/Open-access/Chest CT Scan.json', 'r') as f:
    original_questions = json.load(f)

# Create lookup dictionaries for image paths
augmented_paths = {q['question_id']: q['image_path'] for q in augmented_questions}
original_paths = {q['question_id']: q['image_path'] for q in original_questions}

# Read both JSON files
with open('/home/log/Github/LLaVA-Med/eval_results/chest_ct_results_crop_evaluation.json', 'r') as f:
    file1_data = json.load(f)

with open('/home/log/Github/LLaVA-Med/eval_results/chest_ct_scan/zoom/chest_ct_results2_evaluation.json', 'r') as f:
    file2_data = json.load(f)

# Extract all detailed results from both files
file1_results = []
file2_results = []

for gpu_key in ['gpu0_results', 'gpu1_results']:
    if gpu_key in file1_data and 'detailed_results' in file1_data[gpu_key]:
        file1_results.extend(file1_data[gpu_key]['detailed_results'])
    if gpu_key in file2_data and 'detailed_results' in file2_data[gpu_key]:
        file2_results.extend(file2_data[gpu_key]['detailed_results'])

# Group File 1 results by crop level and original question
file1_by_crop = {}
for r in file1_results:
    crop = r.get('crop', 100)  # Default to 100 if no crop specified
    if crop not in file1_by_crop:
        file1_by_crop[crop] = {}
    
    # Extract base original question ID (remove crop suffixes)
    orig_id = r['original_question_id']
    base_id = orig_id.split('_crop_')[0]  # Get base ID before crop suffix
    
    file1_by_crop[crop][base_id] = r

# Create lookup for File 1 results by base question ID (default crop to 100)  
file1_by_orig = {}
for r in file1_results:
    r_copy = r.copy()
    if 'crop' not in r_copy:
        r_copy['crop'] = 100
    # Extract base ID from file1 too
    orig_id = r['original_question_id'] 
    base_id = orig_id.split('_crop_')[0]
    file1_by_orig[base_id] = r_copy

# Also create lookup for File 2 results by base question ID
file2_by_orig = {}
for r in file2_results:
    r_copy = r.copy()
    if 'crop' not in r_copy:
        r_copy['crop'] = 100
    # File 2 questions are already base IDs (no crop suffix)
    base_id = r['original_question_id']
    file2_by_orig[base_id] = r_copy

# Find all question IDs that have different correctness between any crop level and no crop
questions_with_differences = set()

# Compare all crop levels (file1) with no crop (file2)
for crop_level in file1_by_crop:
    for base_id in file1_by_crop[crop_level]:
        if base_id in file2_by_orig:
            crop_correct = file1_by_crop[crop_level][base_id]['correct']
            no_crop_correct = file2_by_orig[base_id]['correct']
            
            if crop_correct != no_crop_correct:
                questions_with_differences.add(base_id)

print(f"Found {len(questions_with_differences)} questions with differing correctness across crop levels")
print("="*100)

# For each question with differences, show all responses
for i, base_id in enumerate(sorted(questions_with_differences)):
    print(f"\n[{i+1}/{len(questions_with_differences)}] Question: {base_id}")
    print("="*80)
    
    # Get the prompt from any result
    sample_result = None
    if base_id in file2_by_orig:
        sample_result = file2_by_orig[base_id]
    else:
        # Get from any crop level if no crop doesn't exist
        for crop_level in file2_by_crop:
            if base_id in file2_by_crop[crop_level]:
                sample_result = file2_by_crop[crop_level][base_id]
                break
    
    if sample_result:
        print(f"PROMPT: {sample_result['prompt']}")
        print()
    
    # Show no crop result if it exists
    if base_id in file2_by_orig:
        print("NO CROP (File 2):")
        file2_result = file2_by_orig[base_id]
        
        # Get actual image path for no-crop version
        question_id = file2_result['question_id']
        image_path = original_paths.get(question_id, f"Path not found for {question_id}")
        
        print(f"  Image Path: {image_path}")
        print(f"  Correct: {file2_result['correct']}")
        print(f"  GT Answer: {file2_result['gt_answer']}")
        print(f"  Model Response: {file2_result['model_response']}")
        print()
    
    # Show all crop levels for this question
    print("WITH CROP (File 1):")
    for crop_level in sorted([z for z in file1_by_crop.keys() if z is not None]):
        if base_id in file1_by_crop[crop_level]:
            result = file1_by_crop[crop_level][base_id]
            
            # Get actual image path for cropped version
            original_question_id = result['original_question_id']
            image_path = augmented_paths.get(original_question_id, f"Path not found for {original_question_id}")
            
            print(f"  Crop {crop_level}:")
            print(f"    Image Path: {image_path}")
            print(f"    Correct: {result['correct']}")
            print(f"    GT Answer: {result['gt_answer']}")
            print(f"    Model Response: {result['model_response']}")
            print()
    
    print("-" * 80)

print(f"\nProcessed {len(questions_with_differences)} questions with crop-dependent correctness differences.")