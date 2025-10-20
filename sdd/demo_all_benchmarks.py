#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Demo for SDD (Scalable Deadlock Detection) Approach
All Classic Concurrency Benchmarks

This demo showcases the SDD approach across all major concurrency problems:
- DPH – Dining Philosophers
- SHP – Sleeping Barber  
- PLC – Producer–Consumer
- TA – Train Allocation
- FIR – Cigarette Smokers
- RSC – Rail Safety Controller
- BRP – Bridge Crossing Problem
- BTS – Bank Transfer System
- ATSV – Elevator System
"""

import time
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sdd_benchmarks import SDDBenchmark, BenchmarkType, run_all_benchmarks

def demo_individual_benchmarks():
    """Demo each benchmark individually"""
    print("=" * 80)
    print("🎯 Individual SDD Benchmark Demos")
    print("=" * 80)
    
    benchmarks = [
        (BenchmarkType.DPH, "Dining Philosophers", 20),
        (BenchmarkType.SHP, "Sleeping Barber", 15),
        (BenchmarkType.PLC, "Producer-Consumer", 20),
        (BenchmarkType.TA, "Train Allocation", 15),
        (BenchmarkType.FIR, "Cigarette Smokers", 4),
        (BenchmarkType.RSC, "Rail Safety Controller", 12),
        (BenchmarkType.BRP, "Bridge Crossing", 10),
        (BenchmarkType.BTS, "Bank Transfer System", 15),
        (BenchmarkType.ATSV, "Elevator System", 8)
    ]
    
    results = {}
    
    for benchmark_type, name, num_processes in benchmarks:
        print(f"\n{'='*20} {name} {'='*20}")
        
        try:
            # Create benchmark
            benchmark = SDDBenchmark(benchmark_type, num_processes=num_processes)
            
            # Run analysis
            start_time = time.time()
            result = benchmark.run_sdd_analysis(num_iterations=30)
            execution_time = time.time() - start_time
            
            # Store results
            results[benchmark_type.value] = {
                'name': name,
                'deadlocks': result['traditional_deadlocks'],
                'detection_rate': result['performance_metrics']['deadlock_detection_rate'],
                'execution_time': execution_time,
                'severity': result['performance_metrics']['avg_deadlock_severity']
            }
            
            # Quick visualization
            benchmark.visualize_benchmark_state(f"demo_{benchmark_type.value}.png")
            print(f"   📈 Chart saved to demo_{benchmark_type.value}.png")
            
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
            results[benchmark_type.value] = {
                'name': name,
                'deadlocks': 0,
                'detection_rate': 0.0,
                'execution_time': 0.0,
                'severity': 0.0
            }
    
    return results

def demo_benchmark_comparison():
    """Compare all benchmarks"""
    print("\n" + "=" * 80)
    print("⚖️  SDD Benchmark Comparison")
    print("=" * 80)
    
    # Run all benchmarks
    all_results = run_all_benchmarks(num_iterations=30)
    
    # Create comparison charts
    create_comparison_charts(all_results)
    
    return all_results

def create_comparison_charts(all_results):
    """Create comparison charts for all benchmarks"""
    print("\n📊 Creating comparison charts...")
    
    # Extract data
    benchmark_names = []
    deadlock_counts = []
    detection_rates = []
    execution_times = []
    severities = []
    
    for benchmark_type, results in all_results.items():
        benchmark_names.append(benchmark_type.upper())
        deadlock_counts.append(results['traditional_deadlocks'])
        detection_rates.append(results['performance_metrics']['deadlock_detection_rate'])
        execution_times.append(results['execution_time'])
        severities.append(results['performance_metrics']['avg_deadlock_severity'])
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    
    # Deadlock counts
    ax1 = axes[0, 0]
    bars1 = ax1.bar(benchmark_names, deadlock_counts, color='skyblue', alpha=0.7)
    ax1.set_title('Deadlock Counts by Benchmark', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Deadlocks')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars1, deadlock_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom')
    
    # Detection rates
    ax2 = axes[0, 1]
    bars2 = ax2.bar(benchmark_names, detection_rates, color='lightcoral', alpha=0.7)
    ax2.set_title('Deadlock Detection Rates', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Detection Rate')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, rate in zip(bars2, detection_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{rate:.3f}', ha='center', va='bottom')
    
    # Execution times
    ax3 = axes[1, 0]
    bars3 = ax3.bar(benchmark_names, execution_times, color='lightgreen', alpha=0.7)
    ax3.set_title('Execution Times', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Time (seconds)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, time in zip(bars3, execution_times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{time:.2f}s', ha='center', va='bottom')
    
    # Deadlock severity
    ax4 = axes[1, 1]
    bars4 = ax4.bar(benchmark_names, severities, color='gold', alpha=0.7)
    ax4.set_title('Average Deadlock Severity', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Severity Score')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, severity in zip(bars4, severities):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{severity:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('sdd_benchmarks_comparison.png', dpi=300, bbox_inches='tight')
    print("   📈 Comparison chart saved to sdd_benchmarks_comparison.png")
    plt.show()

def demo_scalability_test():
    """Test scalability across different system sizes"""
    print("\n" + "=" * 80)
    print("📈 SDD Scalability Test")
    print("=" * 80)
    
    # Test different system sizes
    system_sizes = [10, 20, 30, 40, 50]
    benchmark_types = [BenchmarkType.DPH, BenchmarkType.PLC, BenchmarkType.BTS]
    
    scalability_results = defaultdict(list)
    
    for benchmark_type in benchmark_types:
        print(f"\nTesting {benchmark_type.value.upper()} scalability...")
        
        for size in system_sizes:
            try:
                benchmark = SDDBenchmark(benchmark_type, num_processes=size)
                start_time = time.time()
                results = benchmark.run_sdd_analysis(num_iterations=20)
                execution_time = time.time() - start_time
                
                scalability_results[benchmark_type.value].append({
                    'size': size,
                    'execution_time': execution_time,
                    'deadlocks': results['traditional_deadlocks'],
                    'detection_rate': results['performance_metrics']['deadlock_detection_rate']
                })
                
                print(f"   Size {size}: {execution_time:.2f}s, {results['traditional_deadlocks']} deadlocks")
                
            except Exception as e:
                print(f"   Size {size}: Error - {e}")
                scalability_results[benchmark_type.value].append({
                    'size': size,
                    'execution_time': 0.0,
                    'deadlocks': 0,
                    'detection_rate': 0.0
                })
    
    # Create scalability chart
    create_scalability_chart(scalability_results)
    
    return scalability_results

def create_scalability_chart(scalability_results):
    """Create scalability analysis chart"""
    print("\n📊 Creating scalability chart...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Execution time vs system size
    ax1 = axes[0]
    for benchmark_type, results in scalability_results.items():
        sizes = [r['size'] for r in results]
        times = [r['execution_time'] for r in results]
        ax1.plot(sizes, times, marker='o', label=benchmark_type.upper(), linewidth=2)
    
    ax1.set_title('Execution Time vs System Size', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Processes')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Deadlock count vs system size
    ax2 = axes[1]
    for benchmark_type, results in scalability_results.items():
        sizes = [r['size'] for r in results]
        deadlocks = [r['deadlocks'] for r in results]
        ax2.plot(sizes, deadlocks, marker='s', label=benchmark_type.upper(), linewidth=2)
    
    ax2.set_title('Deadlock Count vs System Size', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Processes')
    ax2.set_ylabel('Number of Deadlocks')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sdd_scalability_analysis.png', dpi=300, bbox_inches='tight')
    print("   📈 Scalability chart saved to sdd_scalability_analysis.png")
    plt.show()

def demo_performance_analysis():
    """Analyze performance across benchmarks"""
    print("\n" + "=" * 80)
    print("🏁 SDD Performance Analysis")
    print("=" * 80)
    
    # Run comprehensive analysis
    all_results = run_all_benchmarks(num_iterations=50)
    
    # Calculate performance metrics
    performance_metrics = {}
    
    for benchmark_type, results in all_results.items():
        metrics = {
            'efficiency': results['traditional_deadlocks'] / results['execution_time'] if results['execution_time'] > 0 else 0,
            'accuracy': results['performance_metrics']['deadlock_detection_rate'],
            'severity': results['performance_metrics']['avg_deadlock_severity'],
            'speed': 1.0 / results['execution_time'] if results['execution_time'] > 0 else 0,
            'reliability': 1.0 - results['performance_metrics']['avg_deadlock_severity']
        }
        performance_metrics[benchmark_type] = metrics
    
    # Display performance summary
    print(f"\n{'Benchmark':<20} {'Efficiency':<12} {'Accuracy':<12} {'Speed':<12} {'Reliability':<12}")
    print("-" * 80)
    
    for benchmark_type, metrics in performance_metrics.items():
        print(f"{benchmark_type.upper():<20} "
              f"{metrics['efficiency']:<12.3f} "
              f"{metrics['accuracy']:<12.3f} "
              f"{metrics['speed']:<12.3f} "
              f"{metrics['reliability']:<12.3f}")
    
    # Find best performing benchmarks
    best_efficiency = max(performance_metrics.items(), key=lambda x: x[1]['efficiency'])
    best_accuracy = max(performance_metrics.items(), key=lambda x: x[1]['accuracy'])
    best_speed = max(performance_metrics.items(), key=lambda x: x[1]['speed'])
    best_reliability = max(performance_metrics.items(), key=lambda x: x[1]['reliability'])
    
    print(f"\n🏆 Best Performance:")
    print(f"   Efficiency: {best_efficiency[0].upper()} ({best_efficiency[1]['efficiency']:.3f})")
    print(f"   Accuracy: {best_accuracy[0].upper()} ({best_accuracy[1]['accuracy']:.3f})")
    print(f"   Speed: {best_speed[0].upper()} ({best_speed[1]['speed']:.3f})")
    print(f"   Reliability: {best_reliability[0].upper()} ({best_reliability[1]['reliability']:.3f})")
    
    return performance_metrics

def main():
    """Main demo function"""
    print("🎭 SDD (Scalable Deadlock Detection) - All Benchmarks Demo")
    print("Classic Concurrency Problems Analysis")
    print("=" * 80)
    
    try:
        # Individual benchmark demos
        individual_results = demo_individual_benchmarks()
        
        # Benchmark comparison
        comparison_results = demo_benchmark_comparison()
        
        # Scalability test
        scalability_results = demo_scalability_test()
        
        # Performance analysis
        performance_metrics = demo_performance_analysis()
        
        print("\n" + "=" * 80)
        print("✅ All SDD benchmark demos completed successfully!")
        print("📁 Output files:")
        print("   - demo_*.png (individual benchmark visualizations)")
        print("   - sdd_benchmarks_comparison.png")
        print("   - sdd_scalability_analysis.png")
        print("   - sdd_*_results.json (individual results)")
        print("=" * 80)
        
        # Final summary
        total_deadlocks = sum(r['deadlocks'] for r in individual_results.values())
        total_time = sum(r['execution_time'] for r in individual_results.values())
        avg_accuracy = np.mean([r['detection_rate'] for r in individual_results.values()])
        
        print(f"\n📊 Final Summary:")
        print(f"   Total deadlocks detected: {total_deadlocks}")
        print(f"   Total execution time: {total_time:.2f} seconds")
        print(f"   Average accuracy: {avg_accuracy:.3f}")
        print(f"   Benchmarks tested: {len(individual_results)}")
        
    except Exception as e:
        print(f"❌ Error running SDD benchmarks demo: {e}")
        print("Please make sure all required libraries are installed:")
        print("pip install numpy networkx matplotlib")

if __name__ == "__main__":
    main()
