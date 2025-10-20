"""
Enhanced Deadlock Detection Accuracy Evaluation
Gao et al. (2025) dl² methodology with comprehensive accuracy metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import random
from typing import List, Dict, Tuple, Set, Any
import sys
import os
from dataclasses import dataclass, field
import json

# Import dl² methodology
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dl2_deadlock_detector import (
    CommunicationType, OperationState, CommunicationOperation, 
    Process, CommunicationGraph, DL2DeadlockDetector
)


@dataclass
class AccuracyMetrics:
    """Comprehensive accuracy metrics for deadlock detection"""
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    @property
    def f1_score(self) -> float:
        """F1 Score = 2 * (Precision * Recall) / (Precision + Recall)"""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)
    
    @property
    def accuracy(self) -> float:
        """Accuracy = (TP + TN) / (TP + TN + FP + FN)"""
        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total
    
    @property
    def specificity(self) -> float:
        """Specificity = TN / (TN + FP)"""
        if self.true_negatives + self.false_positives == 0:
            return 0.0
        return self.true_negatives / (self.true_negatives + self.false_positives)


class DeadlockTestSuite:
    """
    Comprehensive test suite for evaluating deadlock detection accuracy
    """
    
    def __init__(self):
        self.test_cases: List[Dict[str, Any]] = []
        self.results: List[Dict[str, Any]] = []
        self.overall_metrics = AccuracyMetrics()
        
    def create_test_cases(self) -> List[Dict[str, Any]]:
        """Create comprehensive test cases for deadlock detection"""
        test_cases = []
        
        # Test Case 1: No Deadlock - Simple Communication
        test_cases.append({
            'name': 'No Deadlock - Simple Communication',
            'description': 'Basic communication without deadlock',
            'expected_deadlock': False,
            'philosophers': 10,
            'scenario': 'simple_communication',
            'communication_patterns': [
                {'type': 'broadcast', 'source': 0, 'targets': [1, 2]},
                {'type': 'send', 'source': 3, 'targets': [4]},
                {'type': 'recv', 'source': 5, 'targets': [6]}
            ]
        })
        
        # Test Case 2: Circular Wait Deadlock
        test_cases.append({
            'name': 'Circular Wait Deadlock',
            'description': 'Classic circular wait scenario',
            'expected_deadlock': True,
            'philosophers': 5,
            'scenario': 'circular_wait',
            'communication_patterns': [
                {'type': 'send', 'source': 0, 'targets': [1]},
                {'type': 'send', 'source': 1, 'targets': [2]},
                {'type': 'send', 'source': 2, 'targets': [3]},
                {'type': 'send', 'source': 3, 'targets': [4]},
                {'type': 'send', 'source': 4, 'targets': [0]}
            ]
        })
        
        # Test Case 3: Resource Contention Deadlock
        test_cases.append({
            'name': 'Resource Contention Deadlock',
            'description': 'Multiple processes competing for same resources',
            'expected_deadlock': True,
            'philosophers': 8,
            'scenario': 'resource_contention',
            'communication_patterns': [
                {'type': 'broadcast', 'source': 0, 'targets': [1, 2, 3]},
                {'type': 'broadcast', 'source': 4, 'targets': [5, 6, 7]},
                {'type': 'send', 'source': 1, 'targets': [4]},
                {'type': 'send', 'source': 5, 'targets': [0]}
            ]
        })
        
        # Test Case 4: Complex Communication Pattern
        test_cases.append({
            'name': 'Complex Communication Pattern',
            'description': 'Complex pattern with potential deadlock',
            'expected_deadlock': True,
            'philosophers': 12,
            'scenario': 'complex_pattern',
            'communication_patterns': [
                {'type': 'allreduce', 'source': 0, 'targets': [1, 2, 3]},
                {'type': 'allreduce', 'source': 4, 'targets': [5, 6, 7]},
                {'type': 'broadcast', 'source': 8, 'targets': [9, 10, 11]},
                {'type': 'send', 'source': 1, 'targets': [4]},
                {'type': 'send', 'source': 5, 'targets': [8]},
                {'type': 'send', 'source': 9, 'targets': [0]}
            ]
        })
        
        # Test Case 5: Large Scale System
        test_cases.append({
            'name': 'Large Scale System (100 philosophers)',
            'description': 'Large system with mixed communication patterns',
            'expected_deadlock': True,
            'philosophers': 100,
            'scenario': 'large_scale',
            'communication_patterns': 'auto_generate'
        })
        
        # Test Case 6: False Positive Test
        test_cases.append({
            'name': 'False Positive Test',
            'description': 'Complex communication that should not deadlock',
            'expected_deadlock': False,
            'philosophers': 15,
            'scenario': 'false_positive',
            'communication_patterns': [
                {'type': 'broadcast', 'source': 0, 'targets': [1, 2, 3]},
                {'type': 'broadcast', 'source': 4, 'targets': [5, 6, 7]},
                {'type': 'broadcast', 'source': 8, 'targets': [9, 10, 11]},
                {'type': 'broadcast', 'source': 12, 'targets': [13, 14]},
                {'type': 'send', 'source': 1, 'targets': [4]},
                {'type': 'send', 'source': 5, 'targets': [8]},
                {'type': 'send', 'source': 9, 'targets': [12]},
                {'type': 'send', 'source': 13, 'targets': [0]}
            ]
        })
        
        self.test_cases = test_cases
        return test_cases
    
    def run_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case and return results"""
        print(f"\nRunning test: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print(f"Expected deadlock: {test_case['expected_deadlock']}")
        
        # Create dl² detector
        detector = DL2DeadlockDetector()
        
        # Build job description based on test case
        job_description = self._build_job_description_from_test_case(test_case)
        
        # Run detection
        start_time = time.time()
        result = detector.analyze_deep_learning_job(job_description)
        detection_time = time.time() - start_time
        
        # Determine if detection was correct
        detected_deadlock = result['deadlock_detected']
        expected_deadlock = test_case['expected_deadlock']
        
        # Update accuracy metrics
        if expected_deadlock and detected_deadlock:
            self.overall_metrics.true_positives += 1
            result_type = "True Positive"
        elif expected_deadlock and not detected_deadlock:
            self.overall_metrics.false_negatives += 1
            result_type = "False Negative"
        elif not expected_deadlock and detected_deadlock:
            self.overall_metrics.false_positives += 1
            result_type = "False Positive"
        else:
            self.overall_metrics.true_negatives += 1
            result_type = "True Negative"
        
        test_result = {
            'test_name': test_case['name'],
            'expected_deadlock': expected_deadlock,
            'detected_deadlock': detected_deadlock,
            'result_type': result_type,
            'detection_time': detection_time,
            'analysis_info': result['analysis_info'],
            'recommendations': result['recommendations'],
            'performance_metrics': result['performance_metrics']
        }
        
        print(f"Result: {result_type}")
        print(f"Detection time: {detection_time:.4f}s")
        
        return test_result
    
    def _build_job_description_from_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Build job description from test case"""
        num_philosophers = test_case['philosophers']
        
        # Create processes
        processes = []
        for i in range(num_philosophers):
            processes.append({
                'rank': i,
                'device_id': i % 4,
                'state': 'active'
            })
        
        # Create operations based on communication patterns
        operations = []
        operation_id = 0
        
        if test_case['communication_patterns'] == 'auto_generate':
            # Auto-generate patterns for large scale test
            for i in range(0, num_philosophers, 10):
                operations.append({
                    'id': f'op_{operation_id}',
                    'type': 'broadcast',
                    'source_rank': i,
                    'target_ranks': [(i + j) % num_philosophers for j in range(1, 6)],
                    'tensor_shape': [1024],
                    'state': 'pending'
                })
                operation_id += 1
                
                # Create circular dependency
                if i + 5 < num_philosophers:
                    operations.append({
                        'id': f'op_{operation_id}',
                        'type': 'send',
                        'source_rank': i,
                        'target_ranks': [(i + 5) % num_philosophers],
                        'tensor_shape': [512],
                        'state': 'pending'
                    })
                    operation_id += 1
        else:
            # Use provided patterns
            for pattern in test_case['communication_patterns']:
                operations.append({
                    'id': f'op_{operation_id}',
                    'type': pattern['type'],
                    'source_rank': pattern['source'],
                    'target_ranks': pattern['targets'],
                    'tensor_shape': [256],
                    'state': 'pending'
                })
                operation_id += 1
        
        # Create dependencies for deadlock scenarios
        dependencies = []
        if test_case['expected_deadlock'] and test_case['scenario'] in ['circular_wait', 'resource_contention']:
            for i in range(len(operations) - 1):
                dependencies.append({
                    'operation_id': f'op_{i + 1}',
                    'depends_on': f'op_{i}'
                })
            # Create circular dependency
            if len(operations) > 1:
                dependencies.append({
                    'operation_id': f'op_0',
                    'depends_on': f'op_{len(operations) - 1}'
                })
        
        return {
            'processes': processes,
            'operations': operations,
            'dependencies': dependencies
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test cases and return comprehensive results"""
        print("=" * 80)
        print("COMPREHENSIVE DEADLOCK DETECTION ACCURACY EVALUATION")
        print("Using Gao et al. (2025) dl² Methodology")
        print("=" * 80)
        
        # Create test cases
        test_cases = self.create_test_cases()
        
        # Run each test case
        for test_case in test_cases:
            result = self.run_test_case(test_case)
            self.results.append(result)
        
        # Calculate overall metrics
        overall_results = {
            'total_tests': len(self.results),
            'accuracy_metrics': {
                'accuracy': self.overall_metrics.accuracy,
                'precision': self.overall_metrics.precision,
                'recall': self.overall_metrics.recall,
                'f1_score': self.overall_metrics.f1_score,
                'specificity': self.overall_metrics.specificity,
                'true_positives': self.overall_metrics.true_positives,
                'false_positives': self.overall_metrics.false_positives,
                'true_negatives': self.overall_metrics.true_negatives,
                'false_negatives': self.overall_metrics.false_negatives
            },
            'test_results': self.results,
            'average_detection_time': np.mean([r['detection_time'] for r in self.results])
        }
        
        # Print summary
        self._print_accuracy_summary(overall_results)
        
        return overall_results
    
    def _print_accuracy_summary(self, results: Dict[str, Any]):
        """Print comprehensive accuracy summary"""
        print("\n" + "=" * 80)
        print("ACCURACY EVALUATION SUMMARY")
        print("=" * 80)
        
        metrics = results['accuracy_metrics']
        
        print(f"\nOverall Performance:")
        print(f"  Total Tests: {results['total_tests']}")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"  Recall: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"  F1-Score: {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        print(f"  Specificity: {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
        
        print(f"\nConfusion Matrix:")
        print(f"  True Positives: {metrics['true_positives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  True Negatives: {metrics['true_negatives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        
        print(f"\nPerformance:")
        print(f"  Average Detection Time: {results['average_detection_time']:.4f}s")
        
        print(f"\nDetailed Test Results:")
        for i, result in enumerate(results['test_results'], 1):
            print(f"  {i}. {result['test_name']}: {result['result_type']}")
    
    def create_accuracy_visualization(self, results: Dict[str, Any], save_path: str = None):
        """Create visualization of accuracy results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = results['accuracy_metrics']
        
        # 1. Confusion Matrix
        confusion_matrix = np.array([
            [metrics['true_positives'], metrics['false_positives']],
            [metrics['false_negatives'], metrics['true_negatives']]
        ])
        
        im1 = axes[0, 0].imshow(confusion_matrix, cmap='Blues')
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_xticks([0, 1])
        axes[0, 0].set_yticks([0, 1])
        axes[0, 0].set_xticklabels(['Deadlock', 'No Deadlock'])
        axes[0, 0].set_yticklabels(['Deadlock', 'No Deadlock'])
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                axes[0, 0].text(j, i, confusion_matrix[i, j], 
                              ha='center', va='center', fontsize=14, fontweight='bold')
        
        # 2. Accuracy Metrics Bar Chart
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
        metric_values = [metrics['accuracy'], metrics['precision'], 
                        metrics['recall'], metrics['f1_score'], metrics['specificity']]
        
        bars = axes[0, 1].bar(metric_names, metric_values, color=['blue', 'green', 'orange', 'red', 'purple'])
        axes[0, 1].set_title('Accuracy Metrics')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{value:.3f}', ha='center', va='bottom')
        
        # 3. Test Results Timeline
        test_names = [f"Test {i+1}" for i in range(len(results['test_results']))]
        detection_times = [result['detection_time'] for result in results['test_results']]
        
        axes[1, 0].plot(test_names, detection_times, 'o-', linewidth=2, markersize=8)
        axes[1, 0].set_title('Detection Time by Test Case')
        axes[1, 0].set_xlabel('Test Case')
        axes[1, 0].set_ylabel('Detection Time (s)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Result Type Distribution
        result_types = ['True Positive', 'False Positive', 'True Negative', 'False Negative']
        result_counts = [
            metrics['true_positives'], metrics['false_positives'],
            metrics['true_negatives'], metrics['false_negatives']
        ]
        
        colors = ['green', 'red', 'blue', 'orange']
        wedges, texts, autotexts = axes[1, 1].pie(result_counts, labels=result_types, 
                                                 colors=colors, autopct='%1.1f%%')
        axes[1, 1].set_title('Result Type Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def export_results(self, results: Dict[str, Any], output_path: str):
        """Export results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results exported to {output_path}")


def run_comprehensive_accuracy_evaluation():
    """Run comprehensive accuracy evaluation"""
    print("Starting comprehensive accuracy evaluation...")
    
    # Create test suite
    test_suite = DeadlockTestSuite()
    
    # Run all tests
    results = test_suite.run_all_tests()
    
    # Create visualizations
    print("\nCreating accuracy visualizations...")
    test_suite.create_accuracy_visualization(results, "accuracy_evaluation.png")
    
    # Export results
    test_suite.export_results(results, "accuracy_results.json")
    
    print("\n" + "=" * 80)
    print("ACCURACY EVALUATION COMPLETED!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = run_comprehensive_accuracy_evaluation()
