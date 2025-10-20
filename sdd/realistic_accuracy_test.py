#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realistic SDD Accuracy Test
Tests deadlock detection with more aggressive simulation to generate actual deadlocks
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sdd_benchmarks import SDDBenchmark, BenchmarkType
import json

def create_deadlock_scenario(benchmark_type: BenchmarkType, num_processes: int) -> SDDBenchmark:
    """Create a scenario more likely to generate deadlocks"""
    benchmark = SDDBenchmark(benchmark_type, num_processes=num_processes)
    
    # Force more aggressive resource contention
    if benchmark_type == BenchmarkType.DPH:
        # Force all philosophers to become hungry simultaneously
        for i in range(num_processes):
            benchmark.processes[i].state = "hungry"
            benchmark.graph.nodes[f"philosopher_{i}"]['state'] = "hungry"
    
    elif benchmark_type == BenchmarkType.BTS:
        # Force multiple accounts to be locked simultaneously
        for i in range(min(5, num_processes)):
            benchmark.processes[i].state = "locked"
            benchmark.resources[i].available = False
            benchmark.resources[i].owner = i
            benchmark.graph.nodes[f"account_{i}"]['state'] = "locked"
    
    elif benchmark_type == BenchmarkType.BRP:
        # Force multiple people to try crossing bridge
        for i in range(min(3, num_processes)):
            benchmark.processes[i].state = "crossing"
            benchmark.resources[0].available = False  # Bridge occupied
            benchmark.resources[1].available = False  # Mutex locked
            benchmark.graph.nodes[f"person_{i}"]['state'] = "crossing"
    
    return benchmark

def test_deadlock_detection_accuracy():
    """Test deadlock detection accuracy with realistic scenarios"""
    print("SDD Realistic Deadlock Detection Accuracy Test")
    print("=" * 60)
    
    # Test configurations with higher deadlock probability
    test_configs = [
        (BenchmarkType.DPH, 10, "Dining Philosophers"),
        (BenchmarkType.BTS, 8, "Bank Transfer System"),
        (BenchmarkType.BRP, 6, "Bridge Crossing"),
        (BenchmarkType.TA, 8, "Train Allocation"),
        (BenchmarkType.PLC, 10, "Producer-Consumer")
    ]
    
    results = {}
    
    for benchmark_type, num_processes, name in test_configs:
        print(f"\n{'='*20} {name} {'='*20}")
        
        # Create deadlock-prone scenario
        benchmark = create_deadlock_scenario(benchmark_type, num_processes)
        
        # Run detection
        start_time = time.time()
        is_deadlock, deadlock_info = benchmark._detect_deadlock_traditional()
        detection_time = time.time() - start_time
        
        # Analyze results
        results[benchmark_type.value] = {
            'name': name,
            'num_processes': num_processes,
            'deadlock_detected': is_deadlock,
            'detection_time': detection_time,
            'deadlock_info': deadlock_info,
            'num_cycles': len(deadlock_info.get('cycles', [])),
            'affected_processes': len(deadlock_info.get('affected_processes', [])),
            'severity': deadlock_info.get('severity', 0.0)
        }
        
        print(f"✅ {name} Analysis:")
        print(f"   Deadlock Detected: {'Yes' if is_deadlock else 'No'}")
        print(f"   Detection Time: {detection_time:.4f}s")
        print(f"   Cycles Found: {len(deadlock_info.get('cycles', []))}")
        print(f"   Affected Processes: {len(deadlock_info.get('affected_processes', []))}")
        print(f"   Severity: {deadlock_info.get('severity', 0.0):.3f}")
    
    return results

def test_cycle_detection_accuracy():
    """Test cycle detection accuracy with known deadlock patterns"""
    print("\n" + "=" * 60)
    print("Cycle Detection Accuracy Test")
    print("=" * 60)
    
    # Create artificial deadlock scenarios
    test_cases = []
    
    # Case 1: Simple circular wait
    print("\n🔍 Testing Simple Circular Wait...")
    benchmark1 = SDDBenchmark(BenchmarkType.DPH, num_processes=3)
    # Force circular wait: P0->R0, P1->R1, P2->R2, but P0 needs R1, P1 needs R2, P2 needs R0
    for i in range(3):
        benchmark1.processes[i].state = "hungry"
        benchmark1.resources[i].available = False
        benchmark1.resources[i].owner = i
        benchmark1.processes[i].resources.add(i)
    
    is_deadlock1, info1 = benchmark1._detect_deadlock_traditional()
    test_cases.append({
        'name': 'Simple Circular Wait',
        'expected': True,
        'detected': is_deadlock1,
        'correct': is_deadlock1 == True,
        'cycles': len(info1.get('cycles', [])),
        'severity': info1.get('severity', 0.0)
    })
    
    # Case 2: No deadlock scenario
    print("🔍 Testing No Deadlock Scenario...")
    benchmark2 = SDDBenchmark(BenchmarkType.BTS, num_processes=3)
    # Normal operation - no circular wait
    for i in range(3):
        benchmark2.processes[i].state = "active"
        benchmark2.resources[i].available = True
    
    is_deadlock2, info2 = benchmark2._detect_deadlock_traditional()
    test_cases.append({
        'name': 'No Deadlock',
        'expected': False,
        'detected': is_deadlock2,
        'correct': is_deadlock2 == False,
        'cycles': len(info2.get('cycles', [])),
        'severity': info2.get('severity', 0.0)
    })
    
    # Case 3: Complex deadlock
    print("🔍 Testing Complex Deadlock...")
    benchmark3 = SDDBenchmark(BenchmarkType.BRP, num_processes=4)
    # Multiple processes competing for same resources
    for i in range(4):
        benchmark3.processes[i].state = "crossing"
        if i < 2:  # First two processes hold the bridge
            benchmark3.resources[0].available = False
            benchmark3.resources[1].available = False
    
    is_deadlock3, info3 = benchmark3._detect_deadlock_traditional()
    test_cases.append({
        'name': 'Complex Deadlock',
        'expected': True,
        'detected': is_deadlock3,
        'correct': is_deadlock3 == True,
        'cycles': len(info3.get('cycles', [])),
        'severity': info3.get('severity', 0.0)
    })
    
    return test_cases

def create_accuracy_visualization(deadlock_results, cycle_results):
    """Create visualization of accuracy results"""
    print("\n📊 Creating accuracy visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Deadlock detection results
    ax1 = axes[0, 0]
    benchmark_names = [r['name'] for r in deadlock_results.values()]
    deadlock_detected = [r['deadlock_detected'] for r in deadlock_results.values()]
    colors = ['red' if detected else 'green' for detected in deadlock_detected]
    
    bars1 = ax1.bar(benchmark_names, deadlock_detected, color=colors, alpha=0.7)
    ax1.set_title('Deadlock Detection Results', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Deadlock Detected (1=Yes, 0=No)')
    ax1.set_ylim(0, 1.2)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add labels
    for bar, detected in zip(bars1, deadlock_detected):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                'Yes' if detected else 'No', ha='center', va='bottom')
    
    # Detection times
    ax2 = axes[0, 1]
    detection_times = [r['detection_time'] for r in deadlock_results.values()]
    bars2 = ax2.bar(benchmark_names, detection_times, color='skyblue', alpha=0.7)
    ax2.set_title('Detection Times', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Time (seconds)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, time in zip(bars2, detection_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                f'{time:.4f}s', ha='center', va='bottom')
    
    # Cycle detection accuracy
    ax3 = axes[1, 0]
    test_names = [t['name'] for t in cycle_results]
    correct_detections = [t['correct'] for t in cycle_results]
    colors = ['green' if correct else 'red' for correct in correct_detections]
    
    bars3 = ax3.bar(test_names, correct_detections, color=colors, alpha=0.7)
    ax3.set_title('Cycle Detection Accuracy', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Correct Detection (1=Yes, 0=No)')
    ax3.set_ylim(0, 1.2)
    ax3.tick_params(axis='x', rotation=45)
    
    # Add labels
    for bar, correct in zip(bars3, correct_detections):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                'Correct' if correct else 'Incorrect', ha='center', va='bottom')
    
    # Severity scores
    ax4 = axes[1, 1]
    severities = [r['severity'] for r in deadlock_results.values()]
    bars4 = ax4.bar(benchmark_names, severities, color='orange', alpha=0.7)
    ax4.set_title('Deadlock Severity Scores', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Severity (0-1)')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, severity in zip(bars4, severities):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{severity:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('sdd_realistic_accuracy.png', dpi=300, bbox_inches='tight')
    print("   📈 Realistic accuracy analysis saved to sdd_realistic_accuracy.png")
    plt.show()

def calculate_accuracy_metrics(deadlock_results, cycle_results):
    """Calculate overall accuracy metrics"""
    print("\n📊 Calculating accuracy metrics...")
    
    # Deadlock detection accuracy
    total_deadlock_tests = len(deadlock_results)
    deadlocks_detected = sum(1 for r in deadlock_results.values() if r['deadlock_detected'])
    deadlock_detection_rate = deadlocks_detected / total_deadlock_tests if total_deadlock_tests > 0 else 0
    
    # Cycle detection accuracy
    total_cycle_tests = len(cycle_results)
    correct_cycle_detections = sum(1 for t in cycle_results if t['correct'])
    cycle_detection_accuracy = correct_cycle_detections / total_cycle_tests if total_cycle_tests > 0 else 0
    
    # Average detection time
    avg_detection_time = np.mean([r['detection_time'] for r in deadlock_results.values()])
    
    # Average severity
    avg_severity = np.mean([r['severity'] for r in deadlock_results.values()])
    
    metrics = {
        'deadlock_detection_rate': deadlock_detection_rate,
        'cycle_detection_accuracy': cycle_detection_accuracy,
        'overall_accuracy': (deadlock_detection_rate + cycle_detection_accuracy) / 2,
        'avg_detection_time': avg_detection_time,
        'avg_severity': avg_severity,
        'total_tests': total_deadlock_tests + total_cycle_tests,
        'deadlocks_found': deadlocks_detected,
        'correct_cycles': correct_cycle_detections
    }
    
    return metrics

def main():
    """Main function for realistic accuracy testing"""
    print("SDD (Scalable Deadlock Detection) - Realistic Accuracy Test")
    print("Testing with Deadlock-Prone Scenarios")
    print("=" * 60)
    
    # Test deadlock detection with realistic scenarios
    deadlock_results = test_deadlock_detection_accuracy()
    
    # Test cycle detection accuracy
    cycle_results = test_cycle_detection_accuracy()
    
    # Calculate accuracy metrics
    metrics = calculate_accuracy_metrics(deadlock_results, cycle_results)
    
    # Create visualization
    create_accuracy_visualization(deadlock_results, cycle_results)
    
    # Display results
    print("\n" + "=" * 60)
    print("🎯 REALISTIC ACCURACY ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"📊 Overall Performance:")
    print(f"   Deadlock Detection Rate: {metrics['deadlock_detection_rate']:.3f}")
    print(f"   Cycle Detection Accuracy: {metrics['cycle_detection_accuracy']:.3f}")
    print(f"   Overall Accuracy: {metrics['overall_accuracy']:.3f}")
    print(f"   Average Detection Time: {metrics['avg_detection_time']:.4f}s")
    print(f"   Average Severity: {metrics['avg_severity']:.3f}")
    
    print(f"\n📈 Detailed Results:")
    print(f"   Total Tests: {metrics['total_tests']}")
    print(f"   Deadlocks Found: {metrics['deadlocks_found']}")
    print(f"   Correct Cycle Detections: {metrics['correct_cycles']}")
    
    print(f"\n🔍 Individual Benchmark Results:")
    for benchmark_type, results in deadlock_results.items():
        print(f"   {results['name']:<20} | "
              f"Deadlock: {'Yes' if results['deadlock_detected'] else 'No':<3} | "
              f"Time: {results['detection_time']:.4f}s | "
              f"Severity: {results['severity']:.3f}")
    
    print(f"\n🧪 Cycle Detection Test Results:")
    for test in cycle_results:
        print(f"   {test['name']:<20} | "
              f"Expected: {'Yes' if test['expected'] else 'No':<3} | "
              f"Detected: {'Yes' if test['detected'] else 'No':<3} | "
              f"Correct: {'Yes' if test['correct'] else 'No':<3}")
    
    # Save results
    final_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': metrics,
        'deadlock_results': deadlock_results,
        'cycle_results': cycle_results
    }
    
    with open('sdd_realistic_accuracy_report.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Realistic accuracy analysis completed!")
    print(f"📁 Output files:")
    print(f"   - sdd_realistic_accuracy.png")
    print(f"   - sdd_realistic_accuracy_report.json")

if __name__ == "__main__":
    main()
