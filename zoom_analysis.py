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

print('File 1 (no zoom) vs File 2 (different zoom levels) - Correctness differences:')
print('='*80)

# Find differences between File 1 (no zoom) and each zoom level in File 2
for zoom_level in sorted([z for z in file2_by_zoom.keys() if z is not None]):
    print(f'\nZoom Level {zoom_level}:')
    differences = []
    
    for base_id, file2_result in file2_by_zoom[zoom_level].items():
        if base_id in file1_by_orig:
            file1_correct = file1_by_orig[base_id]['correct']
            file2_correct = file2_result['correct']
            
            if file1_correct != file2_correct:
                differences.append({
                    'base_id': base_id,
                    'file1_correct': file1_correct,
                    'file2_correct': file2_correct,
                    'file2_question_id': file2_result['question_id']
                })
    
    print(f'  Questions with different correctness: {len(differences)}')
    if differences:
        for diff in differences[:5]:  # Show first 5
            print(f'    {diff["base_id"]}: File1={diff["file1_correct"]}, File2={diff["file2_correct"]} (q_id: {diff["file2_question_id"]})')
        if len(differences) > 5:
            print(f'    ... and {len(differences) - 5} more')

# Also show accuracy stats for File 2 by zoom level
print('\n' + '='*80)
print('File 2 Accuracy by Zoom Level:')
for zoom_level in sorted([z for z in file2_by_zoom.keys() if z is not None]):
    results = list(file2_by_zoom[zoom_level].values())
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    print(f'  Zoom {zoom_level}: {accuracy:.4f} ({correct}/{total})')

# Compare File 1 accuracy
file1_correct = sum(1 for r in file1_results if r['correct'])
file1_total = len(file1_results)
file1_accuracy = file1_correct / file1_total if file1_total > 0 else 0
print(f'\nFile 1 (no zoom): {file1_accuracy:.4f} ({file1_correct}/{file1_total})')