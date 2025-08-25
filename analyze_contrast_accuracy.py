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
    
    # Calculate agreement rates between base (1.0) and other contrast levels
    base_contrast = 1.0
    contrast_levels = sorted(contrast_stats.keys())
    base_vs_others_agreement = {}
    
    for contrast_level in contrast_levels:
        if contrast_level != base_contrast:
            pair_key = f"1.0 vs {contrast_level}"
            questions_compared = 0
            questions_agreed = 0
            
            for base_id, contrast_results in all_question_results.items():
                if base_contrast in contrast_results and contrast_level in contrast_results:
                    questions_compared += 1
                    if contrast_results[base_contrast] == contrast_results[contrast_level]:
                        questions_agreed += 1
            
            agreement_rate = questions_agreed / questions_compared if questions_compared > 0 else 0.0
            base_vs_others_agreement[pair_key] = {
                'agreement_rate': agreement_rate,
                'questions_agreed': questions_agreed,
                'questions_compared': questions_compared
            }
    
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
            'total_questions_compared': total_questions,
            'base_vs_others_agreement': base_vs_others_agreement
        }
    }
    
    return summary

def print_analysis_report(analysis):
    """Print a formatted analysis report."""
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
        
        # Print base vs other contrast agreement rates
        if 'base_vs_others_agreement' in agreement and agreement['base_vs_others_agreement']:
            print(f"\nBase (1.0) vs Other Contrast Agreement Rates:")
            print(f"| Contrast Pair          | Agreement | Agreed/Total |")
            print(f"|------------------------|-----------|--------------|")
            
            for pair, stats in agreement['base_vs_others_agreement'].items():
                agreement_rate = stats['agreement_rate']
                agreed = stats['questions_agreed']
                total = stats['questions_compared']
                print(f"| {pair:<20} | {agreement_rate:.4f}    | {agreed:>3}/{total:<8} |")

def main():
    model_types = ["medgemma", "medllava"]   # Hardcoded file paths - modify these to analyze different result files
    datasets = ["chest_ct", "covid19_tianchi"]
    data_types = ["_contrast"]
    # results_files = [
    #     "/home/log/Github/LLaVA-Med/eval_results/medgemma/covid19_tianchi/covid19_tianchi_contrast_evaluation.json",
    #     "/home/log/Github/LLaVA-Med/eval_results/medgemma/covid19_tianchi/covid19_tianchi_evaluation.json"
    # ]
    for model_type in model_types:
        for dataset in datasets:
            for data_type in data_types:
                results_files = [
                    f"/home/log/Github/LLaVA-Med/eval_results/{model_type}/{dataset}/{dataset}{data_type}_evaluation.json",
                    f"/home/log/Github/LLaVA-Med/eval_results/{model_type}/{dataset}/{dataset}_evaluation.json"
                ]
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
                
                
                print("=" * 60)
                print(f"{model_type} {dataset} {data_type} LEVEL ACCURACY ANALYSIS")
                print("=" * 60)
    
                print_analysis_report(analysis)
                print("\n"*5)
    
    return 0

if __name__ == '__main__':
    exit(main())