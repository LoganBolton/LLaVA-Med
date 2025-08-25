#!/usr/bin/env python3
"""
Analyze evaluation results to compute average accuracy per contrast level.
"""
import json
from collections import defaultdict

def analyze_contrast_accuracy(*results_files):
    """
    Analyze the evaluation results and compute accuracy per contrast level.
    Can accept multiple result files to combine analysis.
    
    Args:
        *results_files: One or more paths to evaluation results JSON files
    
    Returns:
        dict: Analysis results including per-contrast accuracy and summary
    """
    # Group results by contrast level
    contrast_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    total_correct_all = 0
    total_count_all = 0
    
    # Store all results for agreement analysis
    all_question_results = defaultdict(dict)  # question_id -> {contrast_level: correctness}
    
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
            contrast = result.get('contrast')
            is_correct = result.get('correct', False)
            
            # Handle null contrast (original images without contrast) - treat as 1.0
            if contrast is None:
                contrast_key = 1.0  # Original images are 100% contrast (1.0)
            else:
                contrast_key = contrast
            
            contrast_stats[contrast_key]['total'] += 1
            
            if is_correct:
                contrast_stats[contrast_key]['correct'] += 1
            
            # Store for agreement analysis - extract base question ID
            original_question_id = result.get('original_question_id', '')
            base_id = original_question_id.split('_contrast_')[0] if '_contrast_' in original_question_id else original_question_id
            all_question_results[base_id][contrast_key] = is_correct
    
    # Calculate overall accuracy
    overall_accuracy = total_correct_all / total_count_all if total_count_all > 0 else 0.0
    
    # Calculate accuracy per contrast level
    contrast_accuracy = {}
    for contrast_level, stats in contrast_stats.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            contrast_accuracy[contrast_level] = {
                'accuracy': accuracy,
                'correct': stats['correct'],
                'total': stats['total']
            }
    
    # Sort contrast levels (numeric order)
    sorted_contrast_accuracy = dict(sorted(contrast_accuracy.items(), key=lambda x: float(x[0])))
    
    # Calculate overall agreement rate (percentage of questions where ALL contrast levels agree)
    total_questions = 0
    questions_with_full_agreement = 0
    
    for base_id, contrast_results in all_question_results.items():
        if len(contrast_results) > 1:  # Only consider questions that appear in multiple contrast levels
            total_questions += 1
            # Check if all contrast levels agree (all True or all False)
            correctness_values = list(contrast_results.values())
            if len(set(correctness_values)) == 1:  # All values are the same
                questions_with_full_agreement += 1
    
    overall_agreement_rate = questions_with_full_agreement / total_questions if total_questions > 0 else 0.0
    
    # Generate summary statistics
    summary = {
        'total_accuracy': overall_accuracy,
        'total_correct': total_correct_all,
        'total_count': total_count_all,
        'contrast_levels_count': len(contrast_accuracy),
        'contrast_accuracy': sorted_contrast_accuracy,
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
    print("contrast LEVEL ACCURACY ANALYSIS")
    print("=" * 60)
    
    print(f"\nOverall Results:")
    print(f"  Total Accuracy: {analysis['total_accuracy']:.4f}")
    print(f"  Total Correct:  {analysis['total_correct']}/{analysis['total_count']}")
    print(f"  contrast Levels:    {analysis['contrast_levels_count']}")
    
    print(f"\n| contrast Level | Accuracy | Correct | Total |")
    print(f"|------------|----------|---------|-------|")
    
    for contrast_level, stats in analysis['contrast_accuracy'].items():
        contrast_str = str(contrast_level) if contrast_level != 'original' else 'original'
        print(f"| {contrast_str:<10} | {stats['accuracy']:.4f}   | {stats['correct']:<7} | {stats['total']:<5} |")
    
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
        "/home/log/Github/LLaVA-Med/eval_results/medgemma/chest_ct_contrast_evaluation.json",
        "/home/log/Github/LLaVA-Med/eval_results/medgemma/chest_ct_evaluation.json"
    ]
    
    # Analyze results
    try:
        analysis = analyze_contrast_accuracy(*results_files)
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