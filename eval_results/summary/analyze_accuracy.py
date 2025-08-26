#!/usr/bin/env python3
"""
Analyze evaluation results to compute average accuracy per contrast level.
"""
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def analyze_contrast_accuracy(*results_files, attribute_type='contrast'):
    """
    Analyze the evaluation results and compute accuracy per contrast/crop level.
    Can accept multiple result files to combine analysis.
    
    Args:
        *results_files: One or more paths to evaluation results JSON files
        attribute_type: 'contrast' or 'crop' - which attribute to analyze
    
    Returns:
        dict: Analysis results including per-attribute accuracy and summary
    """
    # Group results by attribute level (contrast or crop)
    attribute_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    total_correct_all = 0
    total_count_all = 0
    
    # Store all results for agreement analysis
    all_question_results = defaultdict(dict)  # question_id -> {attribute_level: correctness}
    
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
            attribute_value = result.get(attribute_type)
            is_correct = result.get('correct', False)
            
            # Handle null attribute (original images) - treat as base value
            if attribute_value is None:
                if attribute_type == 'contrast':
                    attribute_key = 1.0  # Original images are 100% contrast (1.0)
                elif attribute_type == 'crop':
                    attribute_key = 1.0  # Original images are 100% crop (1.0)
                else:
                    attribute_key = 1.0  # Default base value
            else:
                attribute_key = attribute_value
            
            attribute_stats[attribute_key]['total'] += 1
            
            if is_correct:
                attribute_stats[attribute_key]['correct'] += 1
            
            # Store for agreement analysis - extract base question ID
            original_question_id = result.get('original_question_id', result.get('question_id', ''))
            split_pattern = f'_{attribute_type}_'
            base_id = original_question_id.split(split_pattern)[0] if split_pattern in original_question_id else original_question_id
            all_question_results[base_id][attribute_key] = is_correct
    
    # Calculate overall accuracy
    overall_accuracy = total_correct_all / total_count_all if total_count_all > 0 else 0.0
    
    # Calculate accuracy per attribute level
    attribute_accuracy = {}
    for attribute_level, stats in attribute_stats.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            attribute_accuracy[attribute_level] = {
                'accuracy': accuracy,
                'correct': stats['correct'],
                'total': stats['total']
            }
    
    # Sort attribute levels (numeric order)
    sorted_attribute_accuracy = dict(sorted(attribute_accuracy.items(), key=lambda x: float(x[0])))
    
    # Calculate agreement rates between base (1.0) and other attribute levels
    base_value = 1.0
    attribute_levels = sorted(attribute_stats.keys())
    base_vs_others_agreement = {}
    
    for attribute_level in attribute_levels:
        if attribute_level != base_value:
            pair_key = f"1.0 vs {attribute_level}"
            questions_compared = 0
            questions_agreed = 0
            
            for base_id, attribute_results in all_question_results.items():
                if base_value in attribute_results and attribute_level in attribute_results:
                    questions_compared += 1
                    if attribute_results[base_value] == attribute_results[attribute_level]:
                        questions_agreed += 1
            
            agreement_rate = questions_agreed / questions_compared if questions_compared > 0 else 0.0
            base_vs_others_agreement[pair_key] = {
                'agreement_rate': agreement_rate,
                'questions_agreed': questions_agreed,
                'questions_compared': questions_compared
            }
    
    # Calculate overall agreement rate (percentage of questions where ALL attribute levels agree)
    total_questions = 0
    questions_with_full_agreement = 0
    
    for base_id, attribute_results in all_question_results.items():
        if len(attribute_results) > 1:  # Only consider questions that appear in multiple attribute levels
            total_questions += 1
            # Check if all attribute levels agree (all True or all False)
            correctness_values = list(attribute_results.values())
            if len(set(correctness_values)) == 1:  # All values are the same
                questions_with_full_agreement += 1
    
    overall_agreement_rate = questions_with_full_agreement / total_questions if total_questions > 0 else 0.0
    
    # Generate summary statistics
    summary = {
        'total_accuracy': overall_accuracy,
        'total_correct': total_correct_all,
        'total_count': total_count_all,
        'attribute_levels_count': len(attribute_accuracy),
        'attribute_accuracy': sorted_attribute_accuracy,
        'attribute_type': attribute_type,
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
    attribute_type = analysis.get('attribute_type', 'contrast')
    attribute_type_title = attribute_type.capitalize()
    
    print(f"\nOverall Results:")
    print(f"  Total Accuracy: {analysis['total_accuracy']:.4f}")
    print(f"  Total Correct:  {analysis['total_correct']}/{analysis['total_count']}")
    print(f"  {attribute_type_title} Levels:    {analysis['attribute_levels_count']}")
    
    print(f"\n| {attribute_type_title} Level | Accuracy | Correct | Total |")
    print(f"|------------|----------|---------|-------|")
    
    for attribute_level, stats in analysis['attribute_accuracy'].items():
        attribute_str = str(attribute_level) if attribute_level != 'original' else 'original'
        print(f"| {attribute_str:<10} | {stats['accuracy']:.4f}   | {stats['correct']:<7} | {stats['total']:<5} |")
    
    # Print agreement analysis
    if 'agreement_analysis' in analysis:
        agreement = analysis['agreement_analysis']
        print(f"\nUnanimous Agreement Rate Across Settings: {agreement['overall_agreement_rate']:.4f} ({agreement['overall_agreement_rate']*100:.2f}%)")
        
        # Print base vs other attribute agreement rates
        if 'base_vs_others_agreement' in agreement and agreement['base_vs_others_agreement']:
            attribute_type = analysis.get('attribute_type', 'contrast')
            attribute_type_title = attribute_type.capitalize()
            print(f"\nBase (1.0) vs Other {attribute_type_title} Agreement Rates:")
            print(f"| {attribute_type_title} Pair          | Agreement | Agreed/Total |")
            print(f"|------------------------|-----------|--------------|")
            
            for pair, stats in agreement['base_vs_others_agreement'].items():
                agreement_rate = stats['agreement_rate']
                agreed = stats['questions_agreed']
                total = stats['questions_compared']
                print(f"| {pair:<20} | {agreement_rate:.4f}    | {agreed:>3}/{total:<8} |")

def calculate_weighted_averages(all_results):
    """Calculate weighted averages across datasets and print summary."""
    # Group by model_type and data_type
    grouped_results = {}
    dataset_sizes = {}
    
    for key, analysis in all_results.items():
        model_type, dataset, data_type = key
        attribute_type = data_type[1:]  # Remove underscore
        
        if (model_type, attribute_type) not in grouped_results:
            grouped_results[(model_type, attribute_type)] = []
            dataset_sizes[(model_type, attribute_type)] = []
        
        grouped_results[(model_type, attribute_type)].append(analysis)
        # Use total questions from base dataset (1.0 level) as dataset size
        base_level_total = analysis['attribute_accuracy'].get(1.0, {}).get('total', 0)
        dataset_sizes[(model_type, attribute_type)].append(base_level_total)
    
    print("=" * 80)
    print(f"WEIGHTED AVERAGE RESULTS (chest_ct: 871 samples, covid19_tianchi: 96 samples)")
    print("=" * 80)
    
    # Print all accuracy tables first
    for attribute_type in ['crop', 'contrast']:
        # Accuracy Table
        print(f"\n{attribute_type.capitalize()} Accuracy (Weighted Average)")
        print(f"{attribute_type.capitalize()} Level | MedGemma | MedLLaVA")
        print("-" * 15 + "|" + "-" * 10 + "|" + "-" * 10)
        
        # Get all unique attribute levels across all results
        all_levels = set()
        for (model_type, attr_type), analyses in grouped_results.items():
            if attr_type == attribute_type:
                for analysis in analyses:
                    all_levels.update(analysis['attribute_accuracy'].keys())
        
        sorted_levels = sorted(all_levels, key=float)
        
        for level in sorted_levels:
            row = f"{level:<14} |"
            for model_type in ['medgemma', 'medllava']:
                if (model_type, attribute_type) in grouped_results:
                    analyses = grouped_results[(model_type, attribute_type)]
                    
                    total_correct = 0
                    total_count = 0
                    
                    for analysis in analyses:
                        if level in analysis['attribute_accuracy']:
                            stats = analysis['attribute_accuracy'][level]
                            total_correct += stats['correct']
                            total_count += stats['total']
                    
                    weighted_avg = total_correct / total_count if total_count > 0 else 0.0
                    row += f" {weighted_avg:.4f}   |"
                else:
                    row += "    N/A   |"
            print(row)
    
    # Print all agreement tables second
    for attribute_type in ['crop', 'contrast']:
        # Agreement Rates Table
        print(f"\n{attribute_type.capitalize()} Agreement Rates (Base 1.0 vs Other Levels, Weighted Average)")
        print(f"{attribute_type.capitalize()} Pair      | MedGemma | MedLLaVA")
        print("-" * 15 + "|" + "-" * 10 + "|" + "-" * 10)
        
        # Get all unique agreement pairs
        all_pairs = set()
        for (model_type, attr_type), analyses in grouped_results.items():
            if attr_type == attribute_type:
                for analysis in analyses:
                    if 'agreement_analysis' in analysis and 'base_vs_others_agreement' in analysis['agreement_analysis']:
                        all_pairs.update(analysis['agreement_analysis']['base_vs_others_agreement'].keys())
        
        sorted_pairs = sorted(all_pairs, key=lambda x: float(x.split(' vs ')[1]))
        
        for pair in sorted_pairs:
            row = f"{pair:<14} |"
            for model_type in ['medgemma', 'medllava']:
                if (model_type, attribute_type) in grouped_results:
                    analyses = grouped_results[(model_type, attribute_type)]
                    
                    total_agreed = 0
                    total_compared = 0
                    
                    for analysis in analyses:
                        agreement_analysis = analysis.get('agreement_analysis', {})
                        base_vs_others = agreement_analysis.get('base_vs_others_agreement', {})
                        if pair in base_vs_others:
                            stats = base_vs_others[pair]
                            total_agreed += stats['questions_agreed']
                            total_compared += stats['questions_compared']
                    
                    weighted_agreement = total_agreed / total_compared if total_compared > 0 else 0.0
                    row += f" {weighted_agreement:.4f}   |"
                else:
                    row += "    N/A   |"
            print(row)
    
    print("\n")

def generate_charts(all_results, output_dir="eval_results/summary"):
    """Generate matplotlib charts and save to output directory."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Group results by model_type and data_type
    grouped_results = {}
    
    for key, analysis in all_results.items():
        model_type, dataset, data_type = key
        attribute_type = data_type[1:]  # Remove underscore
        
        if (model_type, attribute_type) not in grouped_results:
            grouped_results[(model_type, attribute_type)] = []
        
        grouped_results[(model_type, attribute_type)].append(analysis)
    
    # Generate charts for each attribute type
    for attribute_type in ['crop', 'contrast']:
        # Accuracy Chart
        plt.figure(figsize=(12, 8))
        
        # Get all unique attribute levels
        all_levels = set()
        for (model_type, attr_type), analyses in grouped_results.items():
            if attr_type == attribute_type:
                for analysis in analyses:
                    all_levels.update(analysis['attribute_accuracy'].keys())
        
        sorted_levels = sorted(all_levels, key=float)
        x_positions = np.arange(len(sorted_levels))
        bar_width = 0.35
        
        # Prepare data for plotting
        medgemma_accuracies = []
        medllava_accuracies = []
        
        for level in sorted_levels:
            # Calculate weighted average for each model
            for model_type in ['medgemma', 'medllava']:
                if (model_type, attribute_type) in grouped_results:
                    analyses = grouped_results[(model_type, attribute_type)]
                    
                    total_correct = 0
                    total_count = 0
                    
                    for analysis in analyses:
                        if level in analysis['attribute_accuracy']:
                            stats = analysis['attribute_accuracy'][level]
                            total_correct += stats['correct']
                            total_count += stats['total']
                    
                    weighted_avg = total_correct / total_count if total_count > 0 else 0.0
                    
                    if model_type == 'medgemma':
                        medgemma_accuracies.append(weighted_avg)
                    else:
                        medllava_accuracies.append(weighted_avg)
        
        # Create bar chart
        plt.bar(x_positions - bar_width/2, medgemma_accuracies, bar_width, 
                label='MedGemma', alpha=0.8, color='#2E86AB')
        plt.bar(x_positions + bar_width/2, medllava_accuracies, bar_width, 
                label='MedLLaVA', alpha=0.8, color='#A23B72')
        
        plt.xlabel(f'{attribute_type.capitalize()} Level')
        plt.ylabel('Accuracy')
        plt.title(f'{attribute_type.capitalize()} Accuracy Comparison (Combined Datasets)')
        plt.xticks(x_positions, [str(level) for level in sorted_levels])
        plt.ylim(0, 0.6)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save accuracy chart
        accuracy_chart_path = os.path.join(output_dir, f'{attribute_type}_accuracy_comparison.png')
        plt.savefig(accuracy_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Agreement Chart
        plt.figure(figsize=(12, 8))
        
        # Get all unique agreement pairs
        all_pairs = set()
        for (model_type, attr_type), analyses in grouped_results.items():
            if attr_type == attribute_type:
                for analysis in analyses:
                    if 'agreement_analysis' in analysis and 'base_vs_others_agreement' in analysis['agreement_analysis']:
                        all_pairs.update(analysis['agreement_analysis']['base_vs_others_agreement'].keys())
        
        sorted_pairs = sorted(all_pairs, key=lambda x: float(x.split(' vs ')[1]))
        x_positions = np.arange(len(sorted_pairs))
        
        # Prepare data for plotting
        medgemma_agreements = []
        medllava_agreements = []
        
        for pair in sorted_pairs:
            for model_type in ['medgemma', 'medllava']:
                if (model_type, attribute_type) in grouped_results:
                    analyses = grouped_results[(model_type, attribute_type)]
                    
                    total_agreed = 0
                    total_compared = 0
                    
                    for analysis in analyses:
                        agreement_analysis = analysis.get('agreement_analysis', {})
                        base_vs_others = agreement_analysis.get('base_vs_others_agreement', {})
                        if pair in base_vs_others:
                            stats = base_vs_others[pair]
                            total_agreed += stats['questions_agreed']
                            total_compared += stats['questions_compared']
                    
                    weighted_agreement = total_agreed / total_compared if total_compared > 0 else 0.0
                    
                    if model_type == 'medgemma':
                        medgemma_agreements.append(weighted_agreement)
                    else:
                        medllava_agreements.append(weighted_agreement)
        
        # Convert to percentage for display
        medgemma_agreements_pct = [x * 100 for x in medgemma_agreements]
        medllava_agreements_pct = [x * 100 for x in medllava_agreements]
        
        # Create bar chart
        plt.bar(x_positions - bar_width/2, medgemma_agreements_pct, bar_width, 
                label='MedGemma', alpha=0.8, color='#2E86AB')
        plt.bar(x_positions + bar_width/2, medllava_agreements_pct, bar_width, 
                label='MedLLaVA', alpha=0.8, color='#A23B72')
        
        plt.xlabel(f'{attribute_type.capitalize()} Comparison')
        plt.ylabel('Agreement Rate (%)')
        plt.title(f'{attribute_type.capitalize()} Agreement Rates (Original vs Augmentation)')
        plt.xticks(x_positions, sorted_pairs, rotation=45)
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save agreement chart
        agreement_chart_path = os.path.join(output_dir, f'{attribute_type}_agreement_comparison.png')
        plt.savefig(agreement_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {attribute_type} charts to {output_dir}/")
    
    print(f"All charts saved to: {output_dir}/")

    print("\n")

def main():
    model_types = ["medgemma", "medllava"]
    datasets = ["chest_ct", "covid19_tianchi"]
    data_types = ["_contrast", "_crop"]
    
    all_results = {}  # Store all analysis results for weighted average calculation
    
    for model_type in model_types:
        for dataset in datasets:
            for data_type in data_types:
                attribute_type = data_type[1:]  # Remove the underscore to get 'contrast' or 'crop'
                results_files = [
                    f"/home/log/Github/LLaVA-Med/eval_results/{model_type}/{dataset}/{dataset}{data_type}_evaluation.json",
                    f"/home/log/Github/LLaVA-Med/eval_results/{model_type}/{dataset}/{dataset}_evaluation.json"
                ]
                
                try:
                    analysis = analyze_contrast_accuracy(*results_files, attribute_type=attribute_type)
                    all_results[(model_type, dataset, data_type)] = analysis
                except FileNotFoundError as e:
                    print(f"Error: File not found: {e}")
                    continue  # Continue with next iteration instead of returning
                except json.JSONDecodeError as e:
                    print(f"Error: Invalid JSON in file: {e}")
                    continue
                except Exception as e:
                    print(f"Error analyzing results: {e}")
                    continue
                
                print("=" * 60)
                print(f"{model_type} {dataset} {data_type} LEVEL ACCURACY ANALYSIS")
                print("=" * 60)
    
                print_analysis_report(analysis)
                print("\n"*5)
    
    # Calculate and print weighted averages
    calculate_weighted_averages(all_results)
    
    # Generate charts
    generate_charts(all_results)
    
    return 0

if __name__ == '__main__':
    exit(main())