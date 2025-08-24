#!/usr/bin/env python3
"""
Analyze evaluation results to compute average accuracy per zoom level.
"""
import json
from collections import defaultdict

def analyze_zoom_accuracy(*results_files):
    """
    Analyze the evaluation results and compute accuracy per zoom level.
    Can accept multiple result files to combine analysis.
    
    Args:
        *results_files: One or more paths to evaluation results JSON files
    
    Returns:
        dict: Analysis results including per-zoom accuracy and summary
    """
    # Group results by zoom level
    zoom_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    total_correct_all = 0
    total_count_all = 0
    
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
            zoom = result.get('zoom')
            is_correct = result.get('correct', False)
            
            # Handle null zoom (original images without zoom) - treat as 1.0
            if zoom is None:
                zoom_key = 1.0  # Original images are 100% zoom (1.0)
            else:
                zoom_key = zoom
            
            zoom_stats[zoom_key]['total'] += 1
            
            if is_correct:
                zoom_stats[zoom_key]['correct'] += 1
    
    # Calculate overall accuracy
    overall_accuracy = total_correct_all / total_count_all if total_count_all > 0 else 0.0
    
    # Calculate accuracy per zoom level
    zoom_accuracy = {}
    for zoom_level, stats in zoom_stats.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            zoom_accuracy[zoom_level] = {
                'accuracy': accuracy,
                'correct': stats['correct'],
                'total': stats['total']
            }
    
    # Sort zoom levels (numeric order)
    sorted_zoom_accuracy = dict(sorted(zoom_accuracy.items(), key=lambda x: float(x[0])))
    
    # Generate summary statistics
    summary = {
        'total_accuracy': overall_accuracy,
        'total_correct': total_correct_all,
        'total_count': total_count_all,
        'zoom_levels_count': len(zoom_accuracy),
        'zoom_accuracy': sorted_zoom_accuracy
    }
    
    return summary

def print_analysis_report(analysis):
    """Print a formatted analysis report."""
    print("=" * 60)
    print("ZOOM LEVEL ACCURACY ANALYSIS")
    print("=" * 60)
    
    print(f"\nOverall Results:")
    print(f"  Total Accuracy: {analysis['total_accuracy']:.4f}")
    print(f"  Total Correct:  {analysis['total_correct']}/{analysis['total_count']}")
    print(f"  Zoom Levels:    {analysis['zoom_levels_count']}")
    
    print(f"\n| Zoom Level | Accuracy | Correct | Total |")
    print(f"|------------|----------|---------|-------|")
    
    for zoom_level, stats in analysis['zoom_accuracy'].items():
        zoom_str = str(zoom_level) if zoom_level != 'original' else 'original'
        print(f"| {zoom_str:<10} | {stats['accuracy']:.4f}   | {stats['correct']:<7} | {stats['total']:<5} |")
    
    # Find best and worst performing zoom levels
    if len(analysis['zoom_accuracy']) > 1:
        best_zoom = max(analysis['zoom_accuracy'].items(), key=lambda x: x[1]['accuracy'])
        worst_zoom = min(analysis['zoom_accuracy'].items(), key=lambda x: x[1]['accuracy'])
        
        print(f"\nPerformance Summary:")
        print(f"  Best:  {best_zoom[0]} ({best_zoom[1]['accuracy']:.4f})")
        print(f"  Worst: {worst_zoom[0]} ({worst_zoom[1]['accuracy']:.4f})")
        
        # Calculate performance difference
        diff = best_zoom[1]['accuracy'] - worst_zoom[1]['accuracy']
        print(f"  Difference: {diff:.4f} ({diff*100:.2f}%)")

def main():
    # Hardcoded file paths - modify these to analyze different result files
    results_files = [
        "eval_results/covid19_tianchi_augmented_evaluation.json",
        "eval_results/covid19_tianchi_evaluation.json"
    ]
    
    # Analyze results
    try:
        analysis = analyze_zoom_accuracy(*results_files)
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