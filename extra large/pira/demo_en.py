#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo file showing how to use TPMC simulators
"""

import time
from tpmc_dining_philosophers_en import TPMCSimulator
from tpmc_advanced_simulation_en import AdvancedTPMCSimulator

def demo_basic_tpmc():
    """Basic TPMC demo"""
    print("=" * 60)
    print("🎯 Basic TPMC Demo")
    print("=" * 60)
    
    # Create simulator
    simulator = TPMCSimulator(num_philosophers=50)  # Fewer for demo
    
    # Run simulation
    start_time = time.time()
    results = simulator.run_tpmc_simulation()
    end_time = time.time()
    
    # Display results
    print(f"\n📊 Basic Version Results:")
    print(f"   ⏱️  Execution time: {end_time - start_time:.2f} seconds")
    print(f"   🔄 Number of iterations: {results['total_iterations']}")
    print(f"   🔍 States explored: {results['states_explored']}")
    print(f"   ⚠️  Deadlocks found: {results['deadlocks_found']}")
    
    # Visualization
    simulator.visualize_system_state("demo_basic_results.png")
    print("   📈 Chart saved to demo_basic_results.png")
    
    return simulator

def demo_advanced_tpmc():
    """Advanced TPMC demo"""
    print("\n" + "=" * 60)
    print("🚀 Advanced TPMC Demo")
    print("=" * 60)
    
    # Create advanced simulator
    simulator = AdvancedTPMCSimulator(
        num_philosophers=50,  # Fewer for demo
        deadlock_probability=0.5  # Higher probability for demo
    )
    
    # Run simulation
    start_time = time.time()
    results = simulator.run_advanced_tpmc_simulation()
    end_time = time.time()
    
    # Display results
    print(f"\n📊 Advanced Version Results:")
    print(f"   ⏱️  Execution time: {end_time - start_time:.2f} seconds")
    print(f"   🔄 Number of iterations: {results['total_iterations']}")
    print(f"   🔍 States explored: {results['states_explored']}")
    print(f"   ⚠️  Deadlocks found: {results['deadlocks_found']}")
    
    if results['performance_metrics']:
        print(f"   📊 Stability score: {results['performance_metrics']['stability_score']:.3f}")
        print(f"   📊 Average resource contention: {results['performance_metrics']['avg_resource_contention']:.3f}")
    
    # Display deadlock details
    if results['deadlocks_found'] > 0:
        print(f"\n🔍 Deadlock Details:")
        for i, deadlock in enumerate(results['deadlock_states'][:3]):  # Show first 3
            print(f"   Deadlock {i+1}:")
            print(f"      Iteration: {deadlock['iteration']}")
            print(f"      Type: {deadlock['deadlock_info']['type']}")
            print(f"      Severity: {deadlock['deadlock_info']['severity']:.3f}")
            print(f"      Cycles: {len(deadlock['deadlock_info']['cycles'])}")
            print(f"      Waiting chains: {len(deadlock['deadlock_info']['waiting_chains'])}")
    
    # Advanced visualization
    simulator.visualize_advanced_system("demo_advanced_results.png")
    print("   📈 Advanced chart saved to demo_advanced_results.png")
    
    # Export results
    simulator.export_advanced_results("demo_advanced_results.json")
    print("   💾 Results saved to demo_advanced_results.json")
    
    return simulator

def demo_comparison():
    """Comparison demo of both versions"""
    print("\n" + "=" * 60)
    print("⚖️  Basic vs Advanced Version Comparison")
    print("=" * 60)
    
    # Test with different philosopher counts
    philosopher_counts = [20, 30, 50]
    
    print(f"{'Philosophers':<15} {'Basic (seconds)':<20} {'Advanced (seconds)':<25}")
    print("-" * 60)
    
    for count in philosopher_counts:
        # Basic version
        basic_sim = TPMCSimulator(num_philosophers=count)
        start_time = time.time()
        basic_sim.run_tpmc_simulation()
        basic_time = time.time() - start_time
        
        # Advanced version
        advanced_sim = AdvancedTPMCSimulator(
            num_philosophers=count,
            deadlock_probability=0.3
        )
        start_time = time.time()
        advanced_sim.run_advanced_tpmc_simulation()
        advanced_time = time.time() - start_time
        
        print(f"{count:<15} {basic_time:<20.2f} {advanced_time:<25.2f}")

def demo_custom_parameters():
    """Demo with custom parameters"""
    print("\n" + "=" * 60)
    print("⚙️  Custom Parameters Demo")
    print("=" * 60)
    
    # Test with different deadlock probabilities
    probabilities = [0.1, 0.3, 0.5, 0.7]
    
    print(f"{'Deadlock Probability':<20} {'Deadlocks Found':<25} {'Execution Time (s)':<20}")
    print("-" * 65)
    
    for prob in probabilities:
        simulator = AdvancedTPMCSimulator(
            num_philosophers=30,
            deadlock_probability=prob
        )
        
        start_time = time.time()
        results = simulator.run_advanced_tpmc_simulation()
        execution_time = time.time() - start_time
        
        print(f"{prob:<20} {results['deadlocks_found']:<25} {execution_time:<20.2f}")

def main():
    """Main demo function"""
    print("🎭 TPMC Simulators Demo")
    print("Based on Mr. Pira's Research")
    print("=" * 60)
    
    try:
        # Basic TPMC demo
        basic_simulator = demo_basic_tpmc()
        
        # Advanced TPMC demo
        advanced_simulator = demo_advanced_tpmc()
        
        # Comparison demo
        demo_comparison()
        
        # Custom parameters demo
        demo_custom_parameters()
        
        print("\n" + "=" * 60)
        print("✅ All demos completed successfully!")
        print("📁 Output files:")
        print("   - demo_basic_results.png")
        print("   - demo_advanced_results.png")
        print("   - demo_advanced_results.json")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        print("Please make sure all required libraries are installed:")
        print("pip install numpy networkx matplotlib")

if __name__ == "__main__":
    main()
