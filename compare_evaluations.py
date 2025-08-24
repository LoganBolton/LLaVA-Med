import json

# Read both JSON files
with open('/home/log/Github/LLaVA-Med/eval_results/chest_ct_results2_evaluation.json', 'r') as f:
    file1_data = json.load(f)

with open('/home/log/Github/LLaVA-Med/eval_results/chest_ct_results_augmented2_evaluation.json', 'r') as f:
    file2_data = json.load(f)

# Extract all detailed results from both files
file1_results = []
file2_results = []

# Combine results from both GPUs
for gpu_key in ['gpu0_results', 'gpu1_results']:
    if gpu_key in file1_data and 'detailed_results' in file1_data[gpu_key]:
        file1_results.extend(file1_data[gpu_key]['detailed_results'])
    if gpu_key in file2_data and 'detailed_results' in file2_data[gpu_key]:
        file2_results.extend(file2_data[gpu_key]['detailed_results'])

print(f"File 1 total results: {len(file1_results)}")
print(f"File 2 total results: {len(file2_results)}")

# Create lookup dictionaries by question_id and original_question_id
file1_by_qid = {r['question_id']: r for r in file1_results}
file1_by_orig_qid = {r['original_question_id']: r for r in file1_results}

file2_by_qid = {r['question_id']: r for r in file2_results}
file2_by_orig_qid = {r['original_question_id']: r for r in file2_results}

# Find differences in correctness by question_id
differences_qid = []
common_qids = set(file1_by_qid.keys()) & set(file2_by_qid.keys())

for qid in common_qids:
    correct1 = file1_by_qid[qid]['correct']
    correct2 = file2_by_qid[qid]['correct']
    
    if correct1 != correct2:
        differences_qid.append({
            'question_id': qid,
            'original_question_id': file1_by_qid[qid]['original_question_id'],
            'file1_correct': correct1,
            'file2_correct': correct2
        })

# Find differences in correctness by original_question_id
differences_orig_qid = []
common_orig_qids = set(file1_by_orig_qid.keys()) & set(file2_by_orig_qid.keys())

for orig_qid in common_orig_qids:
    correct1 = file1_by_orig_qid[orig_qid]['correct']
    correct2 = file2_by_orig_qid[orig_qid]['correct']
    
    if correct1 != correct2:
        differences_orig_qid.append({
            'original_question_id': orig_qid,
            'question_id': file1_by_orig_qid[orig_qid]['question_id'],
            'file1_correct': correct1,
            'file2_correct': correct2
        })

print(f"\nDifferences by question_id: {len(differences_qid)}")
print(f"Differences by original_question_id: {len(differences_orig_qid)}")

print("\nQuestion IDs with different correctness:")
for diff in differences_qid:
    print(f"  {diff['question_id']} (original: {diff['original_question_id']}) - File1: {diff['file1_correct']}, File2: {diff['file2_correct']}")

print("\nOriginal Question IDs with different correctness:")
for diff in differences_orig_qid:
    print(f"  {diff['original_question_id']} (question: {diff['question_id']}) - File1: {diff['file1_correct']}, File2: {diff['file2_correct']}")

# Save all differences to files
with open('/home/log/Github/LLaVA-Med/differences_by_question_id.json', 'w') as f:
    json.dump(differences_qid, f, indent=2)

with open('/home/log/Github/LLaVA-Med/differences_by_original_question_id.json', 'w') as f:
    json.dump(differences_orig_qid, f, indent=2)

print(f"\nSaved all differences to:")
print("- differences_by_question_id.json")
print("- differences_by_original_question_id.json")