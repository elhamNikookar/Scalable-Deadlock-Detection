#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SDD Accuracy Analysis for All Benchmarks
Comprehensive testing to measure deadlock detection accuracy
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sdd_benchmarks import SDDBenchmark, BenchmarkType
import json

def analyze_benchmark_accuracy(benchmark_type: BenchmarkType, num_processes: int, num_iterations: int = 200) -> dict:
    """Analyze accuracy for a specific benchmark"""
    print(f"Analyzing {benchmark_type.value.upper()} accuracy...")
    
    benchmark = SDDBenchmark(benchmark_type, num_processes=num_processes)
    
    # Run multiple trials for better accuracy measurement
    trials = 5
    all_results = []
    
    for trial in range(trials):
        print(f"  Trial {trial + 1}/{trials}")
        
        # Reset benchmark for each trial
        benchmark = SDDBenchmark(benchmark_type, num_processes=num_processes)
        
        # Run analysis
        start_time = time.time()
        results = benchmark.run_sdd_analysis(num_iterations=num_iterations)
        execution_time = time.time() - start_time
        
        all_results.append({
            'trial': trial + 1,
            'deadlocks': results['traditional_deadlocks'],
            'detection_rate': results['performance_metrics']['deadlock_detection_rate'],
            'execution_time': execution_time,
            'severity': results['performance_metrics']['avg_deadlock_severity'],
            'graph_density': results['performance_metrics']['graph_density']
        })
    
    # Calculate accuracy metrics
    deadlock_counts = [r['deadlocks'] for r in all_results]
    detection_rates = [r['detection_rate'] for r in all_results]
    execution_times = [r['execution_time'] for r in all_results]
    severities = [r['severity'] for r in all_results]
    
    accuracy_metrics = {
        'benchmark_type': benchmark_type.value,
        'num_processes': num_processes,
        'num_iterations': num_iterations,
        'trials': trials,
        'avg_deadlocks': np.mean(deadlock_counts),
        'std_deadlocks': np.std(deadlock_counts),
        'avg_detection_rate': np.mean(detection_rates),
        'std_detection_rate': np.std(detection_rates),
        'avg_execution_time': np.mean(execution_times),
        'std_execution_time': np.std(execution_times),
        'avg_severity': np.mean(severities),
        'std_severity': np.std(severities),
        'min_detection_rate': np.min(detection_rates),
        'max_detection_rate': np.max(detection_rates),
        'consistency': 1.0 - (np.std(detection_rates) / (np.mean(detection_rates) + 1e-6)),
        'all_trials': all_results
    }
    
    return accuracy_metrics

def run_comprehensive_accuracy_analysis():
    """Run comprehensive accuracy analysis for all benchmarks"""
    print("SDD Accuracy Analysis - All Benchmarks")
    print("=" * 60)
    
    # Define test configurations
    test_configs = [
        (BenchmarkType.DPH, 20, 100),   # Dining Philosophers
        (BenchmarkType.SHP, 15, 100),   # Sleeping Barber
        (BenchmarkType.PLC, 20, 100),   # Producer-Consumer
        (BenchmarkType.TA, 15, 100),    # Train Allocation
        (BenchmarkType.FIR, 4, 100),    # Cigarette Smokers
        (BenchmarkType.RSC, 12, 100),   # Rail Safety Controller
        (BenchmarkType.BRP, 10, 100),   # Bridge Crossing
        (BenchmarkType.BTS, 15, 100),   # Bank Transfer
        (BenchmarkType.ATSV, 8, 100)    # Elevator System
    ]
    
    all_accuracy_results = {}
    
    for benchmark_type, num_processes, num_iterations in test_configs:
        print(f"\n{'='*20} {benchmark_type.value.upper()} {'='*20}")
        
        try:
            accuracy_metrics = analyze_benchmark_accuracy(
                benchmark_type, num_processes, num_iterations
            )
            all_accuracy_results[benchmark_type.value] = accuracy_metrics
            
            # Display results
            print(f"✅ {benchmark_type.value.upper()} Analysis Complete:")
            print(f"   Average Deadlocks: {accuracy_metrics['avg_deadlocks']:.2f} ± {accuracy_metrics['std_deadlocks']:.2f}")
            print(f"   Detection Rate: {accuracy_metrics['avg_detection_rate']:.3f} ± {accuracy_metrics['std_detection_rate']:.3f}")
            print(f"   Execution Time: {accuracy_metrics['avg_execution_time']:.3f}s ± {accuracy_metrics['std_execution_time']:.3f}s")
            print(f"   Consistency: {accuracy_metrics['consistency']:.3f}")
            print(f"   Min/Max Detection Rate: {accuracy_metrics['min_detection_rate']:.3f} / {accuracy_metrics['max_detection_rate']:.3f}")
            
        except Exception as e:
            print(f"❌ Error analyzing {benchmark_type.value}: {e}")
            all_accuracy_results[benchmark_type.value] = {
                'benchmark_type': benchmark_type.value,
                'error': str(e),
                'avg_detection_rate': 0.0,
                'consistency': 0.0
            }
    
    return all_accuracy_results

def create_accuracy_visualization(accuracy_results):
    """Create visualization of accuracy results"""
    print("\n📊 Creating accuracy visualization...")
    
    # Extract data
    benchmark_names = []
    detection_rates = []
    detection_errors = []
    consistencies = []
    execution_times = []
    
    for benchmark_type, results in accuracy_results.items():
        if 'error' not in results:
            benchmark_names.append(benchmark_type.upper())
            detection_rates.append(results['avg_detection_rate'])
            detection_errors.append(results['std_detection_rate'])
            consistencies.append(results['consistency'])
            execution_times.append(results['avg_execution_time'])
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    
    # Detection rates with error bars
    ax1 = axes[0, 0]
    bars1 = ax1.bar(benchmark_names, detection_rates, yerr=detection_errors, 
                   color='skyblue', alpha=0.7, capsize=5)
    ax1.set_title('Deadlock Detection Rates with Error Bars', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Detection Rate')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, rate, error in zip(bars1, detection_rates, detection_errors):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + error + 0.01, 
                f'{rate:.3f}±{error:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Consistency scores
    ax2 = axes[0, 1]
    bars2 = ax2.bar(benchmark_names, consistencies, color='lightcoral', alpha=0.7)
    ax2.set_title('Detection Consistency Scores', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Consistency (1.0 = Perfect)')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, consistency in zip(bars2, consistencies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{consistency:.3f}', ha='center', va='bottom')
    
    # Execution times
    ax3 = axes[1, 0]
    bars3 = ax3.bar(benchmark_names, execution_times, color='lightgreen', alpha=0.7)
    ax3.set_title('Average Execution Times', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Time (seconds)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, time in zip(bars3, execution_times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{time:.3f}s', ha='center', va='bottom')
    
    # Accuracy summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate summary statistics
    avg_detection_rate = np.mean(detection_rates)
    avg_consistency = np.mean(consistencies)
    avg_execution_time = np.mean(execution_times)
    
    summary_text = f"""
SDD Accuracy Analysis Summary:

Overall Performance:
• Average Detection Rate: {avg_detection_rate:.3f}
• Average Consistency: {avg_consistency:.3f}
• Average Execution Time: {avg_execution_time:.3f}s

Best Performers:
• Highest Detection Rate: {max(detection_rates):.3f}
• Most Consistent: {max(consistencies):.3f}
• Fastest: {min(execution_times):.3f}s

Benchmark Count: {len(benchmark_names)}
Total Trials: {sum(5 for _ in accuracy_results.values() if 'error' not in _)}
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('sdd_accuracy_analysis.png', dpi=300, bbox_inches='tight')
    print("   📈 Accuracy analysis saved to sdd_accuracy_analysis.png")
    plt.show()

def create_detailed_accuracy_report(accuracy_results):
    """Create detailed accuracy report"""
    print("\n📋 Creating detailed accuracy report...")
    
    report = {
        'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {},
        'benchmark_details': accuracy_results
    }
    
    # Calculate summary statistics
    valid_results = {k: v for k, v in accuracy_results.items() if 'error' not in v}
    
    if valid_results:
        detection_rates = [r['avg_detection_rate'] for r in valid_results.values()]
        consistencies = [r['consistency'] for r in valid_results.values()]
        execution_times = [r['avg_execution_time'] for r in valid_results.values()]
        
        report['summary'] = {
            'total_benchmarks': len(valid_results),
            'avg_detection_rate': np.mean(detection_rates),
            'std_detection_rate': np.std(detection_rates),
            'min_detection_rate': np.min(detection_rates),
            'max_detection_rate': np.max(detection_rates),
            'avg_consistency': np.mean(consistencies),
            'std_consistency': np.std(consistencies),
            'avg_execution_time': np.mean(execution_times),
            'std_execution_time': np.std(execution_times),
            'best_detection_benchmark': max(valid_results.items(), key=lambda x: x[1]['avg_detection_rate'])[0],
            'most_consistent_benchmark': max(valid_results.items(), key=lambda x: x[1]['consistency'])[0],
            'fastest_benchmark': min(valid_results.items(), key=lambda x: x[1]['avg_execution_time'])[0]
        }
    
    # Save detailed report
    with open('sdd_accuracy_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("   📄 Detailed report saved to sdd_accuracy_report.json")
    
    return report

def main():
    """Main function for accuracy analysis"""
    print("SDD (Scalable Deadlock Detection) - Accuracy Analysis")
    print("Comprehensive Testing Across All Benchmarks")
    print("=" * 60)
    
    # Run comprehensive accuracy analysis
    accuracy_results = run_comprehensive_accuracy_analysis()
    
    # Create visualizations
    create_accuracy_visualization(accuracy_results)
    
    # Create detailed report
    report = create_detailed_accuracy_report(accuracy_results)
    
    # Display final summary
    print("\n" + "=" * 60)
    print("🎯 ACCURACY ANALYSIS SUMMARY")
    print("=" * 60)
    
    if 'summary' in report and report['summary']:
        summary = report['summary']
        print(f"📊 Overall Performance:")
        print(f"   Average Detection Rate: {summary['avg_detection_rate']:.3f} ± {summary['std_detection_rate']:.3f}")
        print(f"   Average Consistency: {summary['avg_consistency']:.3f} ± {summary['std_consistency']:.3f}")
        print(f"   Average Execution Time: {summary['avg_execution_time']:.3f}s ± {summary['std_execution_time']:.3f}s")
        
        print(f"\n🏆 Best Performers:")
        print(f"   Highest Detection Rate: {summary['best_detection_benchmark'].upper()} ({summary['max_detection_rate']:.3f})")
        print(f"   Most Consistent: {summary['most_consistent_benchmark'].upper()} ({summary['max_consistency']:.3f})")
        print(f"   Fastest: {summary['fastest_benchmark'].upper()} ({summary['min_execution_time']:.3f}s)")
        
        print(f"\n📈 Individual Benchmark Results:")
        for benchmark_type, results in accuracy_results.items():
            if 'error' not in results:
                print(f"   {benchmark_type.upper():<20} | "
                      f"Detection: {results['avg_detection_rate']:.3f}±{results['std_detection_rate']:.3f} | "
                      f"Consistency: {results['consistency']:.3f} | "
                      f"Time: {results['avg_execution_time']:.3f}s")
    
    print(f"\n✅ Accuracy analysis completed successfully!")
    print(f"📁 Output files:")
    print(f"   - sdd_accuracy_analysis.png")
    print(f"   - sdd_accuracy_report.json")

if __name__ == "__main__":
    main()
