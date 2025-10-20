"""
Main Demonstration Script for Kaid et al. (2021) Methodology
Comprehensive evaluation of colored resource-oriented Petri nets with neural network control
"""

import sys
import os
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from philosophers_kaid import run_philosophers_kaid_experiment
from accuracy_evaluation import run_comprehensive_kaid_evaluation


def main():
    """Main demonstration function"""
    print("=" * 80)
    print("KAID ET AL. (2021) METHODOLOGY DEMONSTRATION")
    print("Colored Resource-Oriented Petri Nets with Neural Network Control")
    print("Applied to 100 Philosophers Problem")
    print("=" * 80)
    
    try:
        # Run philosophers experiment
        print("Running philosophers experiment with Kaid methodology...")
        eval_results, sim_results = run_philosophers_kaid_experiment()
        
        # Run comprehensive accuracy evaluation
        print("\nRunning comprehensive accuracy evaluation...")
        accuracy_results = run_comprehensive_kaid_evaluation()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Print final summary
        print(f"\nPhilosophers Experiment Results:")
        print(f"  Total Steps: {sim_results['simulation_steps']}")
        print(f"  Deadlock Detections: {sim_results['deadlock_detections']}")
        print(f"  Fault Detections: {sim_results['fault_detections']}")
        print(f"  Control Success Rate: {sim_results['control_success_rate']:.4f}")
        
        print(f"\nAccuracy Evaluation Results:")
        metrics = accuracy_results['performance_metrics']
        print(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"  Overall Precision: {metrics['overall_precision']:.4f}")
        print(f"  Overall Recall: {metrics['overall_recall']:.4f}")
        print(f"  Overall F1-Score: {metrics['overall_f1_score']:.4f}")
        print(f"  Average Detection Time: {metrics['average_detection_time']:.4f}s")
        
        # Method comparison
        comparison = accuracy_results['comparison_results']
        print(f"\nMethod Comparison:")
        print(f"  Best Method: {comparison['best_method']}")
        print(f"  Improvement over Traditional: {comparison['improvement_over_traditional']*100:.1f}%")
        print(f"  Improvement over Graph-based: {comparison['improvement_over_graph_based']*100:.1f}%")
        print(f"  Improvement over ML: {comparison['improvement_over_ml']*100:.1f}%")
        
        print(f"\nGenerated Files:")
        print(f"  - kaid_evaluation_dashboard.html")
        print(f"  - kaid_performance_analysis.html")
        print(f"  - kaid_accuracy_heatmap.html")
        print(f"  - kaid_evaluation_results.json")
        
        # Effectiveness assessment
        overall_accuracy = metrics['overall_accuracy']
        if overall_accuracy >= 0.9:
            effectiveness = "Excellent"
        elif overall_accuracy >= 0.8:
            effectiveness = "Very Good"
        elif overall_accuracy >= 0.7:
            effectiveness = "Good"
        else:
            effectiveness = "Needs Improvement"
        
        print(f"\nMethod Effectiveness: {effectiveness}")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
