#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realistic SDD Accuracy Implementation
Correct accuracy calculation (0-100%) with proper metrics
"""

import numpy as np
import networkx as nx
import time
import random
from typing import Tuple, Dict, List, Set
from sdd_enhanced import EnhancedSDDBenchmark, BenchmarkType
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class RealisticSDDBenchmark(EnhancedSDDBenchmark):
    """Realistic SDD implementation with correct accuracy calculation"""
    
    def __init__(self, benchmark_type: BenchmarkType, num_processes: int = 50):
        super().__init__(benchmark_type, num_processes)
        self.deadlock_probability = 0.4  # Realistic probability
        self.true_deadlocks = []  # Ground truth deadlocks
        self.predicted_deadlocks = []  # Predicted deadlocks
        self.iteration_count = 0
    
    def _create_realistic_deadlock_scenarios(self, num_scenarios: int = 50):
        """Create realistic deadlock scenarios for testing"""
        scenarios = []
        
        for i in range(num_scenarios):
            # Create a scenario with known deadlock potential
            scenario = {
                'id': i,
                'has_deadlock': random.random() < 0.3,  # 30% chance of actual deadlock
                'processes': [],
                'resources': [],
                'expected_deadlock': False
            }
            
            # Generate process states
            for j in range(self.num_processes):
                if scenario['has_deadlock'] and random.random() < 0.6:
                    # Create deadlock-prone state
                    process_state = {
                        'id': j,
                        'state': 'waiting',
                        'wait_time': random.uniform(2.0, 10.0),
                        'deadlock_risk': random.uniform(0.7, 1.0),
                        'contention_level': random.uniform(0.6, 1.0)
                    }
                    scenario['expected_deadlock'] = True
                else:
                    # Create normal state
                    process_state = {
                        'id': j,
                        'state': random.choice(['idle', 'active', 'thinking']),
                        'wait_time': random.uniform(0.0, 2.0),
                        'deadlock_risk': random.uniform(0.0, 0.3),
                        'contention_level': random.uniform(0.0, 0.4)
                    }
                
                scenario['processes'].append(process_state)
            
            # Generate resource states
            for j in range(len(self.resources)):
                if scenario['has_deadlock'] and random.random() < 0.5:
                    # Create resource contention
                    resource_state = {
                        'id': j,
                        'available': False,
                        'request_queue': [random.randint(0, self.num_processes-1) for _ in range(random.randint(2, 4))],
                        'contention_level': random.uniform(0.6, 1.0)
                    }
                else:
                    # Create normal resource state
                    resource_state = {
                        'id': j,
                        'available': random.choice([True, False]),
                        'request_queue': [] if random.random() < 0.8 else [random.randint(0, self.num_processes-1)],
                        'contention_level': random.uniform(0.0, 0.4)
                    }
                
                scenario['resources'].append(resource_state)
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _apply_scenario(self, scenario: Dict):
        """Apply a scenario to the current system state"""
        # Update processes
        for i, process in enumerate(self.processes):
            if i < len(scenario['processes']):
                proc_state = scenario['processes'][i]
                process.state = proc_state['state']
                process.wait_time = proc_state['wait_time']
                process.deadlock_risk = proc_state['deadlock_risk']
                process.contention_level = proc_state['contention_level']
        
        # Update resources
        for i, resource in enumerate(self.resources):
            if i < len(scenario['resources']):
                res_state = scenario['resources'][i]
                resource.available = res_state['available']
                resource.request_queue = res_state['request_queue'].copy()
                resource.contention_level = res_state['contention_level']
    
    def _detect_realistic_deadlock(self) -> Tuple[bool, Dict]:
        """Realistic deadlock detection with proper accuracy calculation"""
        deadlock_info = {
            'type': 'none',
            'cycles': [],
            'waiting_chains': [],
            'resource_contention': 0.0,
            'severity': 0.0,
            'affected_processes': [],
            'detection_methods': [],
            'confidence': 0.0
        }
        
        detection_methods = []
        confidence_scores = []
        
        # Method 1: Cycle detection
        try:
            cycles = list(nx.simple_cycles(self.graph))
            if len(cycles) > 0:
                detection_methods.append('cycle_detection')
                confidence_scores.append(0.9)
                deadlock_info['cycles'] = cycles
        except:
            pass
        
        # Method 2: Waiting process analysis
        waiting_processes = [p for p in self.processes if p.state in ['waiting', 'hungry', 'blocked']]
        if len(waiting_processes) > 2:  # Need at least 3 waiting processes
            detection_methods.append('waiting_analysis')
            confidence_scores.append(0.7)
            deadlock_info['affected_processes'].extend([f"process_{p.id}" for p in waiting_processes])
        
        # Method 3: Resource contention analysis
        contested_resources = [r for r in self.resources if len(r.request_queue) > 1]
        if len(contested_resources) > 0:
            detection_methods.append('resource_contention')
            confidence_scores.append(0.6)
            deadlock_info['resource_contention'] = len(contested_resources) / len(self.resources)
        
        # Method 4: Long wait time analysis
        long_waiting = [p for p in self.processes if p.wait_time > 5.0]
        if len(long_waiting) > 0:
            detection_methods.append('starvation')
            confidence_scores.append(0.8)
            deadlock_info['affected_processes'].extend([f"process_{p.id}" for p in long_waiting])
        
        # Method 5: High risk analysis
        high_risk = [p for p in self.processes if p.deadlock_risk > 0.8]
        if len(high_risk) > 0:
            detection_methods.append('risk_analysis')
            confidence_scores.append(0.5)
            deadlock_info['affected_processes'].extend([f"process_{p.id}" for p in high_risk])
        
        # Determine if deadlock is detected
        is_deadlock = len(detection_methods) >= 2  # Need at least 2 methods to confirm
        
        if is_deadlock:
            deadlock_info['detection_methods'] = detection_methods
            deadlock_info['confidence'] = np.mean(confidence_scores) if confidence_scores else 0.0
            deadlock_info['type'] = 'realistic'
            deadlock_info['severity'] = min(len(detection_methods) / 5.0, 1.0)
        
        return is_deadlock, deadlock_info
    
    def run_realistic_accuracy_test(self, num_scenarios: int = 100) -> Dict:
        """Run realistic accuracy test with proper metrics"""
        print(f"Starting Realistic SDD Accuracy Test for {self.benchmark_type.value.upper()}")
        print(f"System: {self.num_processes} processes, {len(self.resources)} resources")
        print("=" * 60)
        
        # Create realistic scenarios
        scenarios = self._create_realistic_deadlock_scenarios(num_scenarios)
        
        # Ground truth and predictions
        y_true = []
        y_pred = []
        y_pred_proba = []
        
        results = {
            'benchmark_type': self.benchmark_type.value,
            'total_scenarios': num_scenarios,
            'true_deadlocks': 0,
            'predicted_deadlocks': 0,
            'correct_predictions': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'true_negatives': 0,
            'execution_time': 0.0,
            'detailed_results': []
        }
        
        start_time = time.time()
        
        for i, scenario in enumerate(scenarios):
            # Apply scenario
            self._apply_scenario(scenario)
            
            # Get ground truth
            has_deadlock = scenario['expected_deadlock']
            y_true.append(1 if has_deadlock else 0)
            
            if has_deadlock:
                results['true_deadlocks'] += 1
            
            # Detect deadlock
            is_deadlock, deadlock_info = self._detect_realistic_deadlock()
            y_pred.append(1 if is_deadlock else 0)
            y_pred_proba.append(deadlock_info['confidence'])
            
            if is_deadlock:
                results['predicted_deadlocks'] += 1
            
            # Calculate correct predictions
            if has_deadlock and is_deadlock:
                results['correct_predictions'] += 1
            elif not has_deadlock and not is_deadlock:
                results['correct_predictions'] += 1
            elif not has_deadlock and is_deadlock:
                results['false_positives'] += 1
            elif has_deadlock and not is_deadlock:
                results['false_negatives'] += 1
            
            # Store detailed results
            result_detail = {
                'scenario_id': i,
                'true_deadlock': has_deadlock,
                'predicted_deadlock': is_deadlock,
                'confidence': deadlock_info['confidence'],
                'detection_methods': deadlock_info['detection_methods'],
                'correct': (has_deadlock == is_deadlock)
            }
            results['detailed_results'].append(result_detail)
            
            # Show progress
            if i % 20 == 0:
                print(f"Scenario {i}/{num_scenarios} - "
                      f"True: {results['true_deadlocks']}, "
                      f"Predicted: {results['predicted_deadlocks']}, "
                      f"Correct: {results['correct_predictions']}")
        
        end_time = time.time()
        results['execution_time'] = end_time - start_time
        
        # Calculate proper accuracy metrics
        if len(y_true) > 0 and len(y_pred) > 0:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        else:
            accuracy = precision = recall = f1 = 0.0
        
        results['metrics'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positive_rate': recall,
            'false_positive_rate': results['false_positives'] / max(results['total_scenarios'] - results['true_deadlocks'], 1),
            'specificity': results['true_negatives'] / max(results['total_scenarios'] - results['true_deadlocks'], 1)
        }
        
        print(f"\n✅ Realistic SDD Accuracy Test completed for {self.benchmark_type.value.upper()}!")
        print(f"⏱️  Execution time: {results['execution_time']:.2f} seconds")
        print(f"📊 Total scenarios: {results['total_scenarios']}")
        print(f"🎯 True deadlocks: {results['true_deadlocks']}")
        print(f"🔍 Predicted deadlocks: {results['predicted_deadlocks']}")
        print(f"✅ Correct predictions: {results['correct_predictions']}")
        print(f"📈 Accuracy: {results['metrics']['accuracy']:.3f}")
        print(f"🎯 Precision: {results['metrics']['precision']:.3f}")
        print(f"🔄 Recall: {results['metrics']['recall']:.3f}")
        print(f"⚖️  F1-Score: {results['metrics']['f1_score']:.3f}")
        
        return results

def run_realistic_accuracy_test_all_benchmarks():
    """Run realistic accuracy test for all benchmarks"""
    print("Realistic SDD Accuracy Test - Proper 0-100% Accuracy Calculation")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        (BenchmarkType.DPH, 20, "Dining Philosophers"),
        (BenchmarkType.SHP, 15, "Sleeping Barber"),
        (BenchmarkType.PLC, 20, "Producer-Consumer"),
        (BenchmarkType.TA, 15, "Train Allocation"),
        (BenchmarkType.FIR, 4, "Cigarette Smokers"),
        (BenchmarkType.RSC, 12, "Rail Safety Controller"),
        (BenchmarkType.BRP, 10, "Bridge Crossing"),
        (BenchmarkType.BTS, 15, "Bank Transfer System"),
        (BenchmarkType.ATSV, 8, "Elevator System")
    ]
    
    all_results = {}
    
    for benchmark_type, num_processes, name in test_configs:
        print(f"\n{'='*20} {name} {'='*20}")
        
        # Create realistic benchmark
        benchmark = RealisticSDDBenchmark(benchmark_type, num_processes=num_processes)
        
        # Run realistic accuracy test
        results = benchmark.run_realistic_accuracy_test(num_scenarios=100)
        
        all_results[benchmark_type.value] = {
            'name': name,
            'num_processes': num_processes,
            'accuracy': results['metrics']['accuracy'],
            'precision': results['metrics']['precision'],
            'recall': results['metrics']['recall'],
            'f1_score': results['metrics']['f1_score'],
            'execution_time': results['execution_time'],
            'true_deadlocks': results['true_deadlocks'],
            'predicted_deadlocks': results['predicted_deadlocks'],
            'correct_predictions': results['correct_predictions']
        }
        
        print(f"✅ {name} Results:")
        print(f"   Accuracy: {results['metrics']['accuracy']:.3f}")
        print(f"   Precision: {results['metrics']['precision']:.3f}")
        print(f"   Recall: {results['metrics']['recall']:.3f}")
        print(f"   F1-Score: {results['metrics']['f1_score']:.3f}")
    
    return all_results

def main():
    """Main function for realistic SDD testing"""
    print("Realistic SDD (Scalable Deadlock Detection) - Proper Accuracy Calculation")
    print("Target: >97% accuracy for each benchmark (0-100% range)")
    print("=" * 60)
    
    # Run realistic accuracy test
    results = run_realistic_accuracy_test_all_benchmarks()
    
    # Calculate overall statistics
    accuracies = [r['accuracy'] for r in results.values()]
    precisions = [r['precision'] for r in results.values()]
    recalls = [r['recall'] for r in results.values()]
    f1_scores = [r['f1_score'] for r in results.values()]
    
    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)
    min_accuracy = np.min(accuracies)
    max_accuracy = np.max(accuracies)
    
    print(f"\n" + "=" * 60)
    print("🎯 REALISTIC SDD ACCURACY RESULTS")
    print("=" * 60)
    
    print(f"📊 Overall Performance:")
    print(f"   Average Accuracy: {avg_accuracy:.3f}")
    print(f"   Average Precision: {avg_precision:.3f}")
    print(f"   Average Recall: {avg_recall:.3f}")
    print(f"   Average F1-Score: {avg_f1:.3f}")
    print(f"   Min Accuracy: {min_accuracy:.3f}")
    print(f"   Max Accuracy: {max_accuracy:.3f}")
    
    print(f"\n📈 Individual Benchmark Results:")
    for benchmark_type, result in results.items():
        status = "✅" if result['accuracy'] >= 0.97 else "⚠️"
        print(f"   {status} {result['name']:<20} | "
              f"Accuracy: {result['accuracy']:.3f} | "
              f"Precision: {result['precision']:.3f} | "
              f"Recall: {result['recall']:.3f} | "
              f"F1: {result['f1_score']:.3f}")
    
    # Check if target achieved
    target_achieved = all(r['accuracy'] >= 0.97 for r in results.values())
    
    print(f"\n🎯 Target Achievement:")
    if target_achieved:
        print(f"   ✅ SUCCESS: All benchmarks achieved >97% accuracy!")
    else:
        print(f"   ⚠️  PARTIAL: Some benchmarks need improvement")
        below_target = [r for r in results.values() if r['accuracy'] < 0.97]
        print(f"   Benchmarks below 97%: {len(below_target)}")
    
    # Save results
    final_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'target_achieved': target_achieved,
        'overall_accuracy': avg_accuracy,
        'overall_precision': avg_precision,
        'overall_recall': avg_recall,
        'overall_f1_score': avg_f1,
        'benchmark_results': results
    }
    
    with open('realistic_sdd_accuracy_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Realistic SDD accuracy test completed!")
    print(f"📁 Results saved to realistic_sdd_accuracy_results.json")

if __name__ == "__main__":
    main()
