#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive SDD Accuracy Analysis
Detailed analysis of deadlock detection capabilities across all benchmarks
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sdd_benchmarks import SDDBenchmark, BenchmarkType
import json
import networkx as nx

def analyze_deadlock_detection_capabilities():
    """Analyze the actual deadlock detection capabilities"""
    print("SDD Comprehensive Accuracy Analysis")
    print("=" * 60)
    
    # Test different scenarios for each benchmark
    test_scenarios = {
        BenchmarkType.DPH: {
            'name': 'Dining Philosophers',
            'scenarios': [
                {'name': 'Normal Operation', 'deadlock_expected': False, 'processes': 5},
                {'name': 'High Contention', 'deadlock_expected': True, 'processes': 10},
                {'name': 'Large System', 'deadlock_expected': False, 'processes': 20}
            ]
        },
        BenchmarkType.BTS: {
            'name': 'Bank Transfer System',
            'scenarios': [
                {'name': 'Normal Operation', 'deadlock_expected': False, 'processes': 5},
                {'name': 'Circular Locks', 'deadlock_expected': True, 'processes': 8},
                {'name': 'Large System', 'deadlock_expected': False, 'processes': 15}
            ]
        },
        BenchmarkType.BRP: {
            'name': 'Bridge Crossing',
            'scenarios': [
                {'name': 'Normal Operation', 'deadlock_expected': False, 'processes': 3},
                {'name': 'High Traffic', 'deadlock_expected': True, 'processes': 6},
                {'name': 'Large System', 'deadlock_expected': False, 'processes': 10}
            ]
        }
    }
    
    all_results = {}
    
    for benchmark_type, config in test_scenarios.items():
        print(f"\n{'='*20} {config['name']} {'='*20}")
        benchmark_results = []
        
        for scenario in config['scenarios']:
            print(f"\n🔍 Testing: {scenario['name']}")
            
            # Create benchmark
            benchmark = SDDBenchmark(benchmark_type, num_processes=scenario['processes'])
            
            # Create deadlock-prone scenario if expected
            if scenario['deadlock_expected']:
                benchmark = create_deadlock_scenario(benchmark, benchmark_type)
            
            # Test detection
            start_time = time.time()
            is_deadlock, deadlock_info = benchmark._detect_deadlock_traditional()
            detection_time = time.time() - start_time
            
            # Analyze graph structure
            graph_features = analyze_graph_structure(benchmark.graph)
            
            result = {
                'scenario_name': scenario['name'],
                'num_processes': scenario['processes'],
                'expected_deadlock': scenario['deadlock_expected'],
                'detected_deadlock': is_deadlock,
                'correct_detection': is_deadlock == scenario['deadlock_expected'],
                'detection_time': detection_time,
                'deadlock_info': deadlock_info,
                'graph_features': graph_features
            }
            
            benchmark_results.append(result)
            
            print(f"   Expected: {'Yes' if scenario['deadlock_expected'] else 'No'}")
            print(f"   Detected: {'Yes' if is_deadlock else 'No'}")
            print(f"   Correct: {'Yes' if result['correct_detection'] else 'No'}")
            print(f"   Time: {detection_time:.4f}s")
            print(f"   Cycles: {len(deadlock_info.get('cycles', []))}")
            print(f"   Graph Density: {graph_features['density']:.3f}")
        
        all_results[benchmark_type.value] = {
            'name': config['name'],
            'results': benchmark_results
        }
    
    return all_results

def create_deadlock_scenario(benchmark, benchmark_type):
    """Create a scenario more likely to have deadlocks"""
    if benchmark_type == BenchmarkType.DPH:
        # Force circular wait: all philosophers hungry, forks in circular dependency
        for i in range(len(benchmark.processes)):
            benchmark.processes[i].state = "hungry"
            benchmark.graph.nodes[f"philosopher_{i}"]['state'] = "hungry"
            
            # Create circular dependency
            left_fork = i
            right_fork = (i + 1) % len(benchmark.processes)
            
            # Philosopher i needs right fork but it's held by philosopher (i+1)
            if i < len(benchmark.processes) - 1:
                benchmark.resources[right_fork].available = False
                benchmark.resources[right_fork].owner = (i + 1) % len(benchmark.processes)
                benchmark.processes[(i + 1) % len(benchmark.processes)].resources.add(right_fork)
    
    elif benchmark_type == BenchmarkType.BTS:
        # Create circular lock scenario
        for i in range(min(4, len(benchmark.processes))):
            next_account = (i + 1) % min(4, len(benchmark.processes))
            benchmark.processes[i].state = "locked"
            benchmark.resources[i].available = False
            benchmark.resources[i].owner = i
            benchmark.processes[i].resources.add(i)
            benchmark.processes[i].waiting_for.add(next_account)
            benchmark.graph.nodes[f"account_{i}"]['state'] = "locked"
    
    elif benchmark_type == BenchmarkType.BRP:
        # Create bridge contention
        for i in range(min(3, len(benchmark.processes))):
            benchmark.processes[i].state = "crossing"
            benchmark.graph.nodes[f"person_{i}"]['state'] = "crossing"
        
        # Bridge is occupied but multiple people want it
        benchmark.resources[0].available = False
        benchmark.resources[1].available = False
    
    return benchmark

def analyze_graph_structure(graph):
    """Analyze the structure of the graph"""
    features = {
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'density': nx.density(graph),
        'is_strongly_connected': nx.is_strongly_connected(graph),
        'num_strongly_connected_components': nx.number_strongly_connected_components(graph),
        'avg_degree': np.mean([d for n, d in graph.degree()]) if graph.number_of_nodes() > 0 else 0,
        'clustering_coefficient': nx.average_clustering(graph.to_undirected()) if graph.number_of_nodes() > 0 else 0
    }
    
    # Find cycles
    try:
        cycles = list(nx.simple_cycles(graph))
        features['num_cycles'] = len(cycles)
        features['cycle_lengths'] = [len(cycle) for cycle in cycles]
    except:
        features['num_cycles'] = 0
        features['cycle_lengths'] = []
    
    return features

def calculate_accuracy_metrics(all_results):
    """Calculate comprehensive accuracy metrics"""
    print("\n📊 Calculating accuracy metrics...")
    
    total_tests = 0
    correct_detections = 0
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0
    
    detection_times = []
    graph_densities = []
    
    for benchmark_type, benchmark_data in all_results.items():
        for result in benchmark_data['results']:
            total_tests += 1
            detection_times.append(result['detection_time'])
            graph_densities.append(result['graph_features']['density'])
            
            if result['correct_detection']:
                correct_detections += 1
                if result['expected_deadlock']:
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                if result['expected_deadlock'] and not result['detected_deadlock']:
                    false_negatives += 1
                elif not result['expected_deadlock'] and result['detected_deadlock']:
                    false_positives += 1
    
    # Calculate metrics
    accuracy = correct_detections / total_tests if total_tests > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    avg_detection_time = np.mean(detection_times)
    avg_graph_density = np.mean(graph_densities)
    
    metrics = {
        'total_tests': total_tests,
        'correct_detections': correct_detections,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'avg_detection_time': avg_detection_time,
        'avg_graph_density': avg_graph_density
    }
    
    return metrics

def create_comprehensive_visualization(all_results, metrics):
    """Create comprehensive visualization of results"""
    print("\n📊 Creating comprehensive visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Accuracy by benchmark
    ax1 = axes[0, 0]
    benchmark_names = []
    accuracies = []
    
    for benchmark_type, benchmark_data in all_results.items():
        correct = sum(1 for r in benchmark_data['results'] if r['correct_detection'])
        total = len(benchmark_data['results'])
        accuracy = correct / total if total > 0 else 0
        
        benchmark_names.append(benchmark_type.upper())
        accuracies.append(accuracy)
    
    bars1 = ax1.bar(benchmark_names, accuracies, color='skyblue', alpha=0.7)
    ax1.set_title('Accuracy by Benchmark', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Detection times
    ax2 = axes[0, 1]
    times = []
    for benchmark_type, benchmark_data in all_results.items():
        for result in benchmark_data['results']:
            times.append(result['detection_time'])
    
    ax2.hist(times, bins=10, color='lightgreen', alpha=0.7, edgecolor='black')
    ax2.set_title('Detection Time Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Frequency')
    
    # Graph density vs accuracy
    ax3 = axes[0, 2]
    densities = []
    correct_flags = []
    
    for benchmark_type, benchmark_data in all_results.items():
        for result in benchmark_data['results']:
            densities.append(result['graph_features']['density'])
            correct_flags.append(1 if result['correct_detection'] else 0)
    
    ax3.scatter(densities, correct_flags, alpha=0.6, color='red')
    ax3.set_title('Graph Density vs Detection Accuracy', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Graph Density')
    ax3.set_ylabel('Correct Detection (1=Yes, 0=No)')
    ax3.set_ylim(-0.1, 1.1)
    
    # Confusion matrix
    ax4 = axes[1, 0]
    confusion_data = [
        [metrics['true_negatives'], metrics['false_positives']],
        [metrics['false_negatives'], metrics['true_positives']]
    ]
    
    im = ax4.imshow(confusion_data, cmap='Blues', alpha=0.7)
    ax4.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Actual')
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(['No Deadlock', 'Deadlock'])
    ax4.set_yticklabels(['No Deadlock', 'Deadlock'])
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax4.text(j, i, str(confusion_data[i][j]), ha='center', va='center', fontweight='bold')
    
    # Performance metrics
    ax5 = axes[1, 1]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]
    
    bars5 = ax5.bar(metric_names, metric_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'], alpha=0.7)
    ax5.set_title('Performance Metrics', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Score')
    ax5.set_ylim(0, 1)
    ax5.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, val in zip(bars5, metric_values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom')
    
    # Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = f"""
Comprehensive SDD Accuracy Analysis

Overall Performance:
• Total Tests: {metrics['total_tests']}
• Accuracy: {metrics['accuracy']:.3f}
• Precision: {metrics['precision']:.3f}
• Recall: {metrics['recall']:.3f}
• F1-Score: {metrics['f1_score']:.3f}

Detection Results:
• True Positives: {metrics['true_positives']}
• True Negatives: {metrics['true_negatives']}
• False Positives: {metrics['false_positives']}
• False Negatives: {metrics['false_negatives']}

Performance:
• Avg Detection Time: {metrics['avg_detection_time']:.4f}s
• Avg Graph Density: {metrics['avg_graph_density']:.3f}
    """
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('sdd_comprehensive_accuracy.png', dpi=300, bbox_inches='tight')
    print("   📈 Comprehensive accuracy analysis saved to sdd_comprehensive_accuracy.png")
    plt.show()

def main():
    """Main function for comprehensive accuracy analysis"""
    print("SDD (Scalable Deadlock Detection) - Comprehensive Accuracy Analysis")
    print("Detailed Analysis of Deadlock Detection Capabilities")
    print("=" * 60)
    
    # Run comprehensive analysis
    all_results = analyze_deadlock_detection_capabilities()
    
    # Calculate metrics
    metrics = calculate_accuracy_metrics(all_results)
    
    # Create visualization
    create_comprehensive_visualization(all_results, metrics)
    
    # Display results
    print("\n" + "=" * 60)
    print("🎯 COMPREHENSIVE ACCURACY ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"📊 Overall Performance:")
    print(f"   Total Tests: {metrics['total_tests']}")
    print(f"   Accuracy: {metrics['accuracy']:.3f}")
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall: {metrics['recall']:.3f}")
    print(f"   F1-Score: {metrics['f1_score']:.3f}")
    
    print(f"\n🔍 Detection Results:")
    print(f"   True Positives: {metrics['true_positives']}")
    print(f"   True Negatives: {metrics['true_negatives']}")
    print(f"   False Positives: {metrics['false_positives']}")
    print(f"   False Negatives: {metrics['false_negatives']}")
    
    print(f"\n⏱️  Performance:")
    print(f"   Average Detection Time: {metrics['avg_detection_time']:.4f}s")
    print(f"   Average Graph Density: {metrics['avg_graph_density']:.3f}")
    
    print(f"\n📈 Individual Benchmark Results:")
    for benchmark_type, benchmark_data in all_results.items():
        correct = sum(1 for r in benchmark_data['results'] if r['correct_detection'])
        total = len(benchmark_data['results'])
        accuracy = correct / total if total > 0 else 0
        
        print(f"   {benchmark_type.upper():<20} | "
              f"Accuracy: {accuracy:.3f} | "
              f"Tests: {total}")
    
    # Save detailed results
    final_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': metrics,
        'detailed_results': all_results
    }
    
    with open('sdd_comprehensive_accuracy_report.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Comprehensive accuracy analysis completed!")
    print(f"📁 Output files:")
    print(f"   - sdd_comprehensive_accuracy.png")
    print(f"   - sdd_comprehensive_accuracy_report.json")

if __name__ == "__main__":
    main()
