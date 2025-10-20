"""
Main Accuracy Evaluation Script
Runs comprehensive deadlock detection accuracy evaluation using dl² methodology
"""

import sys
import os
import time
from accuracy_evaluation import run_comprehensive_accuracy_evaluation


def main():
    """Main function to run accuracy evaluation"""
    print("=" * 80)
    print("dl² DEADLOCK DETECTION ACCURACY EVALUATION")
    print("Gao et al. (2025) Methodology")
    print("=" * 80)
    
    try:
        # Run comprehensive accuracy evaluation
        results = run_comprehensive_accuracy_evaluation()
        
        # Print final summary
        print(f"\nFINAL ACCURACY SUMMARY:")
        print(f"  Overall Accuracy: {results['accuracy_metrics']['accuracy']:.4f}")
        print(f"  Precision: {results['accuracy_metrics']['precision']:.4f}")
        print(f"  Recall: {results['accuracy_metrics']['recall']:.4f}")
        print(f"  F1-Score: {results['accuracy_metrics']['f1_score']:.4f}")
        print(f"  Specificity: {results['accuracy_metrics']['specificity']:.4f}")
        
        # Performance summary
        print(f"\nPERFORMANCE SUMMARY:")
        print(f"  Total Tests: {results['total_tests']}")
        print(f"  Average Detection Time: {results['average_detection_time']:.4f}s")
        
        # Method effectiveness
        accuracy = results['accuracy_metrics']['accuracy']
        if accuracy >= 0.9:
            effectiveness = "Excellent"
        elif accuracy >= 0.8:
            effectiveness = "Good"
        elif accuracy >= 0.7:
            effectiveness = "Fair"
        else:
            effectiveness = "Needs Improvement"
        
        print(f"\nMETHOD EFFECTIVENESS: {effectiveness}")
        
        print(f"\nGenerated Files:")
        print(f"  - accuracy_evaluation.png (visualization)")
        print(f"  - accuracy_results.json (detailed results)")
        
    except Exception as e:
        print(f"Error during accuracy evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
