import json

# Read both JSON files
with open('/home/log/Github/LLaVA-Med/eval_results/chest_ct_results2_evaluation.json', 'r') as f:
    file1_data = json.load(f)

with open('/home/log/Github/LLaVA-Med/eval_results/chest_ct_results_augmented2_evaluation.json', 'r') as f:
    file2_data = json.load(f)

# Extract all detailed results from both files
file1_results = []
file2_results = []

for gpu_key in ['gpu0_results', 'gpu1_results']:
    if gpu_key in file1_data and 'detailed_results' in file1_data[gpu_key]:
        file1_results.extend(file1_data[gpu_key]['detailed_results'])
    if gpu_key in file2_data and 'detailed_results' in file2_data[gpu_key]:
        file2_results.extend(file2_data[gpu_key]['detailed_results'])

# Group File 2 results by zoom level and original question
file2_by_zoom = {}
for r in file2_results:
    zoom = r['zoom']
    if zoom not in file2_by_zoom:
        file2_by_zoom[zoom] = {}
    
    # Extract base original question ID (remove zoom suffixes)
    orig_id = r['original_question_id']
    base_id = orig_id.split('_zoom_')[0]  # Get base ID before zoom suffix
    
    file2_by_zoom[zoom][base_id] = r

# Create lookup for File 1 results by original question ID  
file1_by_orig = {r['original_question_id']: r for r in file1_results}

# Find all question IDs that have different correctness across zoom levels
questions_with_differences = set()

for zoom_level in sorted([z for z in file2_by_zoom.keys() if z is not None]):
    for base_id, file2_result in file2_by_zoom[zoom_level].items():
        if base_id in file1_by_orig:
            file1_correct = file1_by_orig[base_id]['correct']
            file2_correct = file2_result['correct']
            
            if file1_correct != file2_correct:
                questions_with_differences.add(base_id)

print(f"Found {len(questions_with_differences)} questions with differing correctness across zoom levels")
print("="*100)

# For each question with differences, show all responses
for i, base_id in enumerate(sorted(questions_with_differences)):
    print(f"\n[{i+1}/{len(questions_with_differences)}] Question: {base_id}")
    print("="*80)
    
    # Get File 1 result (no zoom)
    file1_result = file1_by_orig[base_id]
    print(f"PROMPT: {file1_result['prompt']}")
    print()
    print("NO ZOOM (File 1):")
    print(f"  Correct: {file1_result['correct']}")
    print(f"  GT Answer: {file1_result['gt_answer']}")
    print(f"  Model Response: {file1_result['model_response']}")
    print()
    
    # Get File 2 results for all zoom levels
    print("WITH ZOOM (File 2):")
    for zoom_level in sorted([z for z in file2_by_zoom.keys() if z is not None]):
        if base_id in file2_by_zoom[zoom_level]:
            result = file2_by_zoom[zoom_level][base_id]
            print(f"  Zoom {zoom_level}:")
            print(f"    Correct: {result['correct']}")
            print(f"    GT Answer: {result['gt_answer']}")
            print(f"    Model Response: {result['model_response']}")
            print()
    
    print("-" * 80)

print(f"\nProcessed {len(questions_with_differences)} questions with zoom-dependent correctness differences.")