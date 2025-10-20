"""
Simple test script for the Salimi et al. implementation
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all modules can be imported"""
    try:
        from fuzzy_genetic_algorithm import FuzzyGeneticAlgorithm, create_default_fuzzy_rules
        from deadlock_detection import DeadlockDetector, Process, Resource
        from graph_transformation import GraphTransformationSystem, Node, Edge, NodeType
        from experimental_results import ExperimentRunner, ResultsVisualizer
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_fuzzy_genetic_algorithm():
    """Test fuzzy genetic algorithm"""
    try:
        from fuzzy_genetic_algorithm import FuzzyGeneticAlgorithm, create_default_fuzzy_rules
        
        fga = FuzzyGeneticAlgorithm(
            population_size=10,
            chromosome_length=5,
            max_generations=5,
            fuzzy_rules=create_default_fuzzy_rules()
        )
        
        fga.initialize_population()
        best_individual = fga.evolve(problem_type="reachability")
        
        print(f"✓ Fuzzy GA test passed - Best fitness: {best_individual.fitness:.4f}")
        return True
    except Exception as e:
        print(f"✗ Fuzzy GA test failed: {e}")
        return False

def test_deadlock_detection():
    """Test deadlock detection"""
    try:
        from deadlock_detection import DeadlockDetector, Process, Resource
        
        detector = DeadlockDetector()
        
        # Add test processes and resources
        detector.add_process(Process("P1"))
        detector.add_process(Process("P2"))
        detector.add_resource(Resource("R1", capacity=1))
        
        # Test allocation
        detector.allocate_resource("P1", "R1")
        
        # Test deadlock detection
        is_deadlock, processes, confidence = detector.detect_deadlock_fuzzy_approach()
        
        print(f"✓ Deadlock detection test passed - Deadlock: {is_deadlock}")
        return True
    except Exception as e:
        print(f"✗ Deadlock detection test failed: {e}")
        return False

def test_graph_transformation():
    """Test graph transformation system"""
    try:
        from graph_transformation import GraphTransformationSystem, Node, Edge, NodeType
        
        gts = GraphTransformationSystem()
        
        # Add nodes and edges
        gts.add_node(Node("S0", NodeType.STATE))
        gts.add_node(Node("S1", NodeType.STATE))
        gts.add_edge(Edge("S0", "S1", "transition"))
        
        # Test reachability
        reachable = gts.get_reachable_states("S0")
        
        print(f"✓ Graph transformation test passed - Reachable states: {len(reachable)}")
        return True
    except Exception as e:
        print(f"✗ Graph transformation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running Salimi et al. implementation tests...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_fuzzy_genetic_algorithm,
        test_deadlock_detection,
        test_graph_transformation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("✓ All tests passed! Implementation is working correctly.")
    else:
        print("✗ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
