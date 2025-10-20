#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo file for SDD (Scalable Deadlock Detection) Approach
Shows both full and simplified versions
"""

import time
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demo_simplified_sdd():
    """Demo the simplified SDD approach"""
    print("=" * 60)
    print("🎯 Simplified SDD Demo")
    print("=" * 60)
    
    try:
        from sdd_simplified import SDDDetector
        
        # Create detector
        detector = SDDDetector(num_processes=30, num_resources=20)
        
        # Run analysis
        start_time = time.time()
        results = detector.run_sdd_analysis(num_iterations=50)
        end_time = time.time()
        
        # Display results
        print(f"\n📊 Simplified SDD Results:")
        print(f"   ⏱️  Execution time: {end_time - start_time:.2f} seconds")
        print(f"   🔄 Number of iterations: {results['total_iterations']}")
        print(f"   🔍 Traditional deadlocks: {results['traditional_deadlocks']}")
        print(f"   🧠 ML deadlocks: {results['ml_deadlocks']}")
        print(f"   📊 ML accuracy: {results['performance_metrics']['ml_accuracy']:.3f}")
        
        # Visualization
        detector.visualize_system_state("demo_simplified_sdd.png")
        print("   📈 Chart saved to demo_simplified_sdd.png")
        
        return detector, results
        
    except ImportError as e:
        print(f"❌ Error importing simplified SDD: {e}")
        return None, None

def demo_full_sdd():
    """Demo the full SDD approach (if PyTorch is available)"""
    print("\n" + "=" * 60)
    print("🚀 Full SDD Demo (with PyTorch)")
    print("=" * 60)
    
    try:
        from sdd_approach import SDDDetector
        
        # Create detector
        detector = SDDDetector(num_processes=30, num_resources=20)
        
        # Run analysis
        start_time = time.time()
        results = detector.run_sdd_analysis(num_iterations=50)
        end_time = time.time()
        
        # Display results
        print(f"\n📊 Full SDD Results:")
        print(f"   ⏱️  Execution time: {end_time - start_time:.2f} seconds")
        print(f"   🔄 Number of iterations: {results['total_iterations']}")
        print(f"   🔍 Traditional deadlocks: {results['traditional_deadlocks']}")
        print(f"   🧠 GNN deadlocks: {results['gnn_deadlocks']}")
        print(f"   📊 GNN accuracy: {results['performance_metrics']['gnn_accuracy']:.3f}")
        
        # Visualization
        detector.visualize_system_state("demo_full_sdd.png")
        print("   📈 Chart saved to demo_full_sdd.png")
        
        return detector, results
        
    except ImportError as e:
        print(f"⚠️  PyTorch not available, skipping full SDD demo: {e}")
        print("   💡 Install PyTorch to run the full version:")
        print("   pip install torch torch-geometric")
        return None, None

def demo_comparison():
    """Compare different SDD approaches"""
    print("\n" + "=" * 60)
    print("⚖️  SDD Approach Comparison")
    print("=" * 60)
    
    # Test with different system sizes
    system_sizes = [(20, 15), (30, 20), (40, 25)]
    
    print(f"{'System Size':<15} {'Simplified (s)':<20} {'Full (s)':<15} {'Simplified ML Acc':<20}")
    print("-" * 70)
    
    for processes, resources in system_sizes:
        # Simplified version
        try:
            from sdd_simplified import SDDDetector as SimplifiedDetector
            detector = SimplifiedDetector(num_processes=processes, num_resources=resources)
            start_time = time.time()
            results = detector.run_sdd_analysis(num_iterations=30)
            simplified_time = time.time() - start_time
            simplified_acc = results['performance_metrics']['ml_accuracy']
        except:
            simplified_time = 0.0
            simplified_acc = 0.0
        
        # Full version
        try:
            from sdd_approach import SDDDetector as FullDetector
            detector = FullDetector(num_processes=processes, num_resources=resources)
            start_time = time.time()
            results = detector.run_sdd_analysis(num_iterations=30)
            full_time = time.time() - start_time
        except:
            full_time = 0.0
        
        print(f"{processes}x{resources:<10} {simplified_time:<20.2f} {full_time:<15.2f} {simplified_acc:<20.3f}")

def demo_benchmark():
    """Benchmark SDD against traditional approaches"""
    print("\n" + "=" * 60)
    print("🏁 SDD Benchmarking")
    print("=" * 60)
    
    try:
        from sdd_simplified import SDDDetector
        
        # Test different deadlock scenarios
        scenarios = [
            {"name": "Low Contention", "processes": 20, "resources": 15},
            {"name": "Medium Contention", "processes": 30, "resources": 20},
            {"name": "High Contention", "processes": 40, "resources": 15},
        ]
        
        print(f"{'Scenario':<20} {'Deadlocks':<15} {'ML Accuracy':<15} {'Time (s)':<15}")
        print("-" * 65)
        
        for scenario in scenarios:
            detector = SDDDetector(
                num_processes=scenario["processes"],
                num_resources=scenario["resources"]
            )
            
            start_time = time.time()
            results = detector.run_sdd_analysis(num_iterations=50)
            execution_time = time.time() - start_time
            
            print(f"{scenario['name']:<20} {results['traditional_deadlocks']:<15} "
                  f"{results['performance_metrics']['ml_accuracy']:<15.3f} {execution_time:<15.2f}")
        
    except Exception as e:
        print(f"❌ Error in benchmarking: {e}")

def main():
    """Main demo function"""
    print("🎭 SDD (Scalable Deadlock Detection) Demo")
    print("Based on Graph Transformation Systems and Machine Learning")
    print("=" * 60)
    
    try:
        # Run simplified SDD demo
        simplified_detector, simplified_results = demo_simplified_sdd()
        
        # Run full SDD demo (if available)
        full_detector, full_results = demo_full_sdd()
        
        # Run comparison
        demo_comparison()
        
        # Run benchmark
        demo_benchmark()
        
        print("\n" + "=" * 60)
        print("✅ All SDD demos completed successfully!")
        print("📁 Output files:")
        print("   - demo_simplified_sdd.png")
        if full_detector:
            print("   - demo_full_sdd.png")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error running SDD demo: {e}")
        print("Please make sure all required libraries are installed:")
        print("pip install numpy networkx matplotlib")

if __name__ == "__main__":
    main()
