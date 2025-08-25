#!/usr/bin/env python3
"""
Analyze evaluation results to compute average accuracy per crop level.
"""
import json
from collections import defaultdict

def analyze_crop_accuracy(*results_files):
    """
    Analyze the evaluation results and compute accuracy per crop level.
    Can accept multiple result files to combine analysis.
    
    Args:
        *results_files: One or more paths to evaluation results JSON files
    
    Returns:
        dict: Analysis results including per-crop accuracy and summary
    """
    # Group results by crop level
    crop_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    total_correct_all = 0
    total_count_all = 0
    
    # Store all results for agreement analysis
    all_question_results = defaultdict(dict)  # question_id -> {crop_level: correctness}
    
    for results_file in results_files:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Collect all detailed results from both GPU processes
        all_results = []
        
        # Handle different result file structures
        if 'gpu0_results' in results and 'gpu1_results' in results:
            # Parallel processing results
            all_results.extend(results['gpu0_results']['detailed_results'])
            all_results.extend(results['gpu1_results']['detailed_results'])
            file_correct = results.get('correct', 0)
            file_count = results.get('total', 0)
        else:
            # Single processing results
            all_results = results.get('detailed_results', [])
            file_correct = results.get('correct', 0)
            file_count = results.get('total', 0)
        
        # Accumulate totals for overall accuracy calculation
        total_correct_all += file_correct
        total_count_all += file_count
        
        for result in all_results:
            crop = result.get('crop')
            is_correct = result.get('correct', False)
            
            # Handle null crop (original images without crop) - treat as 1.0
            if crop is None:
                crop_key = 1.0  # Original images are 100% crop (1.0)
            else:
                crop_key = crop
            
            crop_stats[crop_key]['total'] += 1
            
            if is_correct:
                crop_stats[crop_key]['correct'] += 1
            
            # Store for agreement analysis - extract base question ID
            original_question_id = result.get('original_question_id', '')
            base_id = original_question_id.split('_crop_')[0] if '_crop_' in original_question_id else original_question_id
            all_question_results[base_id][crop_key] = is_correct
    
    # Calculate overall accuracy
    overall_accuracy = total_correct_all / total_count_all if total_count_all > 0 else 0.0
    
    # Calculate accuracy per crop level
    crop_accuracy = {}
    for crop_level, stats in crop_stats.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            crop_accuracy[crop_level] = {
                'accuracy': accuracy,
                'correct': stats['correct'],
                'total': stats['total']
            }
    
    # Sort crop levels (numeric order)
    sorted_crop_accuracy = dict(sorted(crop_accuracy.items(), key=lambda x: float(x[0])))
    
    # Calculate overall agreement rate (percentage of questions where ALL crop levels agree)
    total_questions = 0
    questions_with_full_agreement = 0
    
    for base_id, crop_results in all_question_results.items():
        if len(crop_results) > 1:  # Only consider questions that appear in multiple crop levels
            total_questions += 1
            # Check if all crop levels agree (all True or all False)
            correctness_values = list(crop_results.values())
            if len(set(correctness_values)) == 1:  # All values are the same
                questions_with_full_agreement += 1
    
    overall_agreement_rate = questions_with_full_agreement / total_questions if total_questions > 0 else 0.0
    
    # Generate summary statistics
    summary = {
        'total_accuracy': overall_accuracy,
        'total_correct': total_correct_all,
        'total_count': total_count_all,
        'crop_levels_count': len(crop_accuracy),
        'crop_accuracy': sorted_crop_accuracy,
        'agreement_analysis': {
            'overall_agreement_rate': overall_agreement_rate,
            'questions_with_full_agreement': questions_with_full_agreement,
            'total_questions_compared': total_questions
        }
    }
    
    return summary

def print_analysis_report(analysis):
    """Print a formatted analysis report."""
    print("=" * 60)
    print("crop LEVEL ACCURACY ANALYSIS")
    print("=" * 60)
    
    print(f"\nOverall Results:")
    print(f"  Total Accuracy: {analysis['total_accuracy']:.4f}")
    print(f"  Total Correct:  {analysis['total_correct']}/{analysis['total_count']}")
    print(f"  crop Levels:    {analysis['crop_levels_count']}")
    
    print(f"\n| crop Level | Accuracy | Correct | Total |")
    print(f"|------------|----------|---------|-------|")
    
    for crop_level, stats in analysis['crop_accuracy'].items():
        crop_str = str(crop_level) if crop_level != 'original' else 'original'
        print(f"| {crop_str:<10} | {stats['accuracy']:.4f}   | {stats['correct']:<7} | {stats['total']:<5} |")
    
    # Print agreement analysis
    if 'agreement_analysis' in analysis:
        agreement = analysis['agreement_analysis']
        print(f"\nAgreement Analysis:")
        print(f"  Overall Agreement Rate: {agreement['overall_agreement_rate']:.4f} ({agreement['overall_agreement_rate']*100:.2f}%)")
        print(f"  Questions with Full Agreement: {agreement['questions_with_full_agreement']}/{agreement['total_questions_compared']}")
        disagreement_rate = 1 - agreement['overall_agreement_rate']
        print(f"  Disagreement Rate: {disagreement_rate:.4f} ({disagreement_rate*100:.2f}%)")

def main():
    # Hardcoded file paths - modify these to analyze different result files
    results_files = [
        "/home/log/Github/LLaVA-Med/eval_results/medgemma/chest_ct_crop_evaluation.json",
        "/home/log/Github/LLaVA-Med/eval_results/medgemma/chest_ct_evaluation.json"
    ]
    
    # Analyze results
    try:
        analysis = analyze_crop_accuracy(*results_files)
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file: {e}")
        return 1
    except Exception as e:
        print(f"Error analyzing results: {e}")
        return 1
    
    # Print report
    print_analysis_report(analysis)
    
    return 0

if __name__ == '__main__':
    exit(main())