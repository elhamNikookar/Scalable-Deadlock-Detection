"""
Main Demonstration Script
Salimi et al. (2020) Fuzzy Genetic Algorithm for Reachability and Deadlock Detection
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Dict, Tuple
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fuzzy_genetic_algorithm import FuzzyGeneticAlgorithm, create_default_fuzzy_rules, Individual
from deadlock_detection import DeadlockDetector, Process, Resource, AdvancedDeadlockDetector
from graph_transformation import GraphTransformationSystem, PetriNetSystem, Node, Edge, NodeType
from experimental_results import ExperimentRunner, ResultsVisualizer, run_complete_experiment


def demonstrate_fuzzy_genetic_algorithm():
    """
    Demonstrate the Fuzzy Genetic Algorithm for reachability verification
    """
    print("=" * 60)
    print("FUZZY GENETIC ALGORITHM DEMONSTRATION")
    print("=" * 60)
    
    # Create fuzzy genetic algorithm
    fga = FuzzyGeneticAlgorithm(
        population_size=30,
        chromosome_length=20,
        max_generations=50,
        mutation_rate=0.1,
        crossover_rate=0.8,
        fuzzy_rules=create_default_fuzzy_rules()
    )
    
    print(f"Algorithm Parameters:")
    print(f"  Population Size: {fga.population_size}")
    print(f"  Chromosome Length: {fga.chromosome_length}")
    print(f"  Max Generations: {fga.max_generations}")
    print(f"  Mutation Rate: {fga.mutation_rate}")
    print(f"  Crossover Rate: {fga.crossover_rate}")
    print(f"  Fuzzy Rules: {len(fga.fuzzy_rules)}")
    
    # Run evolution for reachability verification
    print(f"\nRunning evolution for reachability verification...")
    start_time = time.time()
    best_individual = fga.evolve(problem_type="reachability")
    execution_time = time.time() - start_time
    
    print(f"Evolution completed in {execution_time:.2f} seconds")
    print(f"Best fitness: {best_individual.fitness:.4f}")
    print(f"Best chromosome: {best_individual.chromosome[:10]}...")  # Show first 10 genes
    
    # Get convergence metrics
    metrics = fga.get_convergence_metrics()
    print(f"\nConvergence Metrics:")
    print(f"  Final Fitness: {metrics['final_fitness']:.4f}")
    print(f"  Best Fitness: {metrics['best_fitness']:.4f}")
    print(f"  Improvement: {metrics['improvement']:.4f}")
    print(f"  Convergence Rate: {metrics['convergence_rate']:.4f}")
    
    # Plot fitness evolution
    if fga.fitness_history:
        plt.figure(figsize=(10, 6))
        plt.plot(fga.fitness_history)
        plt.title('Fitness Evolution - Fuzzy Genetic Algorithm')
        plt.xlabel('Generation')
        plt.ylabel('Average Fitness')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return best_individual, fga


def demonstrate_deadlock_detection():
    """
    Demonstrate deadlock detection algorithms
    """
    print("\n" + "=" * 60)
    print("DEADLOCK DETECTION DEMONSTRATION")
    print("=" * 60)
    
    # Create deadlock detector
    detector = DeadlockDetector()
    
    # Setup a test system with potential deadlock
    print("Setting up test system...")
    
    # Create processes
    processes = [Process(f"P{i}") for i in range(5)]
    for process in processes:
        detector.add_process(process)
    
    # Create resources
    resources = [Resource(f"R{i}", capacity=1) for i in range(3)]
    for resource in resources:
        detector.add_resource(resource)
    
    print(f"Created {len(processes)} processes and {len(resources)} resources")
    
    # Create a circular wait scenario (potential deadlock)
    print("Creating circular wait scenario...")
    
    # Process 0 allocates R0, requests R1
    detector.allocate_resource("P0", "R0")
    detector.allocate_resource("P0", "R1")
    
    # Process 1 allocates R1, requests R2
    detector.allocate_resource("P1", "R1")
    detector.allocate_resource("P1", "R2")
    
    # Process 2 allocates R2, requests R0
    detector.allocate_resource("P2", "R2")
    detector.allocate_resource("P2", "R0")
    
    print("System state after resource allocation:")
    print(f"  Waiting processes: {len([p for p in detector.processes.values() if p.state == 'waiting'])}")
    
    # Test different deadlock detection methods
    methods = [
        ("Cycle Detection", detector.detect_deadlock_cycle_detection),
        ("Resource Allocation", detector.detect_deadlock_resource_allocation),
        ("Fuzzy Approach", detector.detect_deadlock_fuzzy_approach)
    ]
    
    print(f"\nTesting deadlock detection methods:")
    for method_name, method_func in methods:
        start_time = time.time()
        
        if method_name == "Fuzzy Approach":
            is_deadlock, processes, confidence = method_func()
            print(f"  {method_name}:")
            print(f"    Deadlock detected: {is_deadlock}")
            print(f"    Confidence: {confidence:.4f}")
            print(f"    Deadlocked processes: {processes}")
        else:
            is_deadlock, processes = method_func()
            print(f"  {method_name}:")
            print(f"    Deadlock detected: {is_deadlock}")
            print(f"    Deadlocked processes: {processes}")
        
        execution_time = time.time() - start_time
        print(f"    Execution time: {execution_time:.4f} seconds")
        print()
    
    # Demonstrate deadlock resolution
    print("Demonstrating deadlock resolution...")
    is_deadlock, deadlocked_processes, _ = detector.detect_deadlock_fuzzy_approach()
    
    if is_deadlock:
        print(f"Resolving deadlock by terminating processes: {deadlocked_processes}")
        detector.resolve_deadlock(deadlocked_processes, strategy="terminate")
        
        # Check if deadlock is resolved
        is_deadlock_after, _, _ = detector.detect_deadlock_fuzzy_approach()
        print(f"Deadlock resolved: {not is_deadlock_after}")
    
    return detector


def demonstrate_graph_transformation():
    """
    Demonstrate graph transformation system
    """
    print("\n" + "=" * 60)
    print("GRAPH TRANSFORMATION SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Create graph transformation system
    gts = GraphTransformationSystem()
    
    # Add nodes
    nodes = [
        Node("S0", NodeType.STATE, {"tokens": 1}),
        Node("S1", NodeType.STATE, {"tokens": 0}),
        Node("T1", NodeType.PROCESS),
        Node("R1", NodeType.RESOURCE, {"capacity": 1})
    ]
    
    for node in nodes:
        gts.add_node(node)
    
    # Add edges
    edges = [
        Edge("S0", "T1", "input"),
        Edge("T1", "S1", "output"),
        Edge("R1", "T1", "resource")
    ]
    
    for edge in edges:
        gts.add_edge(edge)
    
    print(f"Created graph transformation system with:")
    print(f"  Nodes: {len(gts.nodes)}")
    print(f"  Edges: {len(gts.edges)}")
    
    # Add transformation rules
    rule1 = {
        "name": "transition_fire",
        "precondition": lambda state: "S0" in state,
        "postcondition": lambda state: state.replace("S0", "S1")
    }
    gts.add_transformation_rule(rule1)
    
    print(f"  Transformation rules: {len(gts.transformation_rules)}")
    
    # Test reachability
    print(f"\nTesting reachability from S0:")
    reachable_states = gts.get_reachable_states("S0", max_depth=5)
    print(f"  Reachable states: {reachable_states}")
    
    # Test deadlock detection
    print(f"\nTesting deadlock detection:")
    is_deadlock, deadlock_cycle = gts.detect_deadlock("S0")
    print(f"  Deadlock detected: {is_deadlock}")
    print(f"  Deadlock cycle: {deadlock_cycle}")
    
    # Demonstrate Petri Net
    print(f"\nDemonstrating Petri Net system:")
    petri_net = PetriNetSystem()
    
    # Add places
    petri_net.add_place("P1", 2)
    petri_net.add_place("P2", 0)
    petri_net.add_place("P3", 1)
    
    # Add transitions
    petri_net.add_transition("T1")
    petri_net.add_transition("T2")
    
    # Add arcs
    petri_net.add_arc("P1", "T1", 1)
    petri_net.add_arc("T1", "P2", 1)
    petri_net.add_arc("P3", "T2", 1)
    petri_net.add_arc("T2", "P1", 1)
    
    print(f"  Places: {list(petri_net.places.keys())}")
    print(f"  Transitions: {list(petri_net.transitions)}")
    
    # Test transition firing
    print(f"  Initial places: {petri_net.places}")
    print(f"  T1 enabled: {petri_net.is_transition_enabled('T1')}")
    print(f"  T2 enabled: {petri_net.is_transition_enabled('T2')}")
    
    if petri_net.is_transition_enabled("T1"):
        petri_net.fire_transition("T1")
        print(f"  After firing T1: {petri_net.places}")
    
    # Test deadlock detection
    is_deadlock, enabled_transitions = petri_net.detect_deadlock()
    print(f"  Deadlock detected: {is_deadlock}")
    print(f"  Enabled transitions: {enabled_transitions}")
    
    return gts, petri_net


def demonstrate_experimental_results():
    """
    Demonstrate experimental results and visualization
    """
    print("\n" + "=" * 60)
    print("EXPERIMENTAL RESULTS DEMONSTRATION")
    print("=" * 60)
    
    print("Running a subset of experiments for demonstration...")
    
    # Create experiment runner
    runner = ExperimentRunner("salimi/results")
    
    # Run a small set of experiments
    print("Running reachability experiments...")
    runner.run_reachability_experiments([10, 20], num_trials=3)
    
    print("Running deadlock detection experiments...")
    runner.run_deadlock_detection_experiments([5, 10], num_trials=2)
    
    print("Running comparative experiments...")
    runner.run_comparative_experiments([10, 20], num_trials=2)
    
    # Save results
    runner.save_results("demo_results.json")
    
    # Create visualizations
    print("Creating visualizations...")
    visualizer = ResultsVisualizer(runner.results)
    
    # Show summary statistics
    summary = visualizer.create_summary_table()
    
    # Create plots
    try:
        visualizer.plot_accuracy_vs_size("salimi/results/demo_accuracy.png")
        visualizer.plot_execution_time_vs_size("salimi/results/demo_execution_time.png")
        print("Plots saved to salimi/results/")
    except Exception as e:
        print(f"Error creating plots: {e}")
    
    return runner.results


def demonstrate_advanced_features():
    """
    Demonstrate advanced features
    """
    print("\n" + "=" * 60)
    print("ADVANCED FEATURES DEMONSTRATION")
    print("=" * 60)
    
    # Advanced deadlock detector with history
    print("Demonstrating advanced deadlock detector with history...")
    advanced_detector = AdvancedDeadlockDetector()
    
    # Setup system
    for i in range(3):
        process = Process(f"P{i}")
        advanced_detector.add_process(process)
    
    for i in range(2):
        resource = Resource(f"R{i}", capacity=1)
        advanced_detector.add_resource(resource)
    
    # Simulate system evolution
    print("Simulating system evolution...")
    for step in range(5):
        # Randomly allocate resources
        process_id = f"P{step % 3}"
        resource_id = f"R{step % 2}"
        
        advanced_detector.allocate_resource(process_id, resource_id)
        
        # Detect deadlock with history
        is_deadlock, processes, confidence = advanced_detector.detect_deadlock_with_history()
        
        print(f"  Step {step}: Deadlock={is_deadlock}, Confidence={confidence:.3f}")
    
    # Predict future deadlock probability
    future_prob = advanced_detector.predict_deadlock_probability()
    print(f"  Predicted future deadlock probability: {future_prob:.3f}")
    
    # Demonstrate fuzzy rule evaluation
    print(f"\nDemonstrating fuzzy rule evaluation...")
    from fuzzy_genetic_algorithm import FuzzyRule, FuzzySet
    
    # Create a fuzzy rule
    rule = FuzzyRule(
        antecedents=[FuzzySet(0.3, 0.5, 0.7)],
        consequent=FuzzySet(0.6, 0.8, 1.0)
    )
    
    # Test with different inputs
    test_inputs = [0.2, 0.4, 0.6, 0.8]
    for input_val in test_inputs:
        firing_strength = rule.evaluate([input_val])
        print(f"  Input: {input_val:.1f}, Firing strength: {firing_strength:.3f}")


def main():
    """
    Main demonstration function
    """
    print("SALIMI ET AL. (2020) METHODOLOGY DEMONSTRATION")
    print("Fuzzy Genetic Algorithm for Reachability and Deadlock Detection")
    print("=" * 80)
    
    try:
        # 1. Demonstrate Fuzzy Genetic Algorithm
        best_individual, fga = demonstrate_fuzzy_genetic_algorithm()
        
        # 2. Demonstrate Deadlock Detection
        detector = demonstrate_deadlock_detection()
        
        # 3. Demonstrate Graph Transformation System
        gts, petri_net = demonstrate_graph_transformation()
        
        # 4. Demonstrate Experimental Results
        results = demonstrate_experimental_results()
        
        # 5. Demonstrate Advanced Features
        demonstrate_advanced_features()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        print(f"\nSummary:")
        print(f"  - Fuzzy GA best fitness: {best_individual.fitness:.4f}")
        print(f"  - Graph transformation nodes: {len(gts.nodes)}")
        print(f"  - Experimental results: {len(results)}")
        print(f"  - All demonstrations completed successfully")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
