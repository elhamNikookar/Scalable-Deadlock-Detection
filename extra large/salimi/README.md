"""
README for Salimi et al. (2020) Implementation
Fuzzy Genetic Algorithm for Reachability and Deadlock Detection
"""

# Salimi et al. (2020) Implementation

This repository contains a Python implementation of the methodology described in:

**Salimi, N., Rafe, V., Tabrizchi, H., Mosavi, A.: Fuzzy genetic algorithm approach for verification of reachability and detection of deadlock in graph transformation systems, 000241–000250 (2020)**

## Overview

This implementation provides:

1. **Fuzzy Genetic Algorithm** for reachability verification in graph transformation systems
2. **Deadlock Detection Algorithms** using multiple approaches (cycle detection, resource allocation, fuzzy logic)
3. **Graph Transformation System** modeling capabilities
4. **Experimental Framework** for testing and evaluation
5. **Visualization Tools** for results analysis

## Files Structure

```
salimi/
├── graph_transformation.py      # Graph transformation system classes
├── fuzzy_genetic_algorithm.py   # Fuzzy genetic algorithm implementation
├── deadlock_detection.py        # Deadlock detection algorithms
├── experimental_results.py      # Experimental framework and visualization
├── main_demo.py                # Main demonstration script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the demonstration:
```bash
python main_demo.py
```

## Key Components

### 1. Fuzzy Genetic Algorithm (`fuzzy_genetic_algorithm.py`)

- **FuzzySet**: Triangular membership functions
- **FuzzyRule**: Fuzzy rules with antecedents and consequents
- **FuzzyGeneticAlgorithm**: Main algorithm class with evolution loop
- **Individual**: Chromosome representation

**Key Features:**
- Fuzzy rule-based fitness evaluation
- Tournament selection
- Single-point crossover
- Gaussian mutation with fuzzy constraints
- Convergence tracking

### 2. Deadlock Detection (`deadlock_detection.py`)

- **DeadlockDetector**: Main deadlock detection class
- **Process**: Process representation with resource allocation
- **Resource**: Resource representation with capacity
- **AdvancedDeadlockDetector**: Enhanced detector with history

**Detection Methods:**
- Cycle detection in wait-for graph
- Resource allocation graph analysis
- Banker's algorithm approach
- Fuzzy logic approach with confidence scoring

### 3. Graph Transformation System (`graph_transformation.py`)

- **GraphTransformationSystem**: Base graph transformation system
- **PetriNetSystem**: Petri net implementation
- **Node**: Node representation with types and properties
- **Edge**: Edge representation with weights and types

**Features:**
- Reachability verification
- Deadlock detection
- Transformation rule application
- Petri net firing semantics

### 4. Experimental Framework (`experimental_results.py`)

- **ExperimentRunner**: Runs experiments and collects results
- **ResultsVisualizer**: Creates visualizations and analysis
- **ExperimentResult**: Data container for results

**Experiment Types:**
- Reachability verification experiments
- Deadlock detection experiments
- Comparative algorithm studies
- Performance analysis

## Usage Examples

### Basic Fuzzy Genetic Algorithm

```python
from fuzzy_genetic_algorithm import FuzzyGeneticAlgorithm, create_default_fuzzy_rules

# Create algorithm
fga = FuzzyGeneticAlgorithm(
    population_size=50,
    chromosome_length=20,
    max_generations=100,
    fuzzy_rules=create_default_fuzzy_rules()
)

# Run evolution
best_individual = fga.evolve(problem_type="reachability")
print(f"Best fitness: {best_individual.fitness}")
```

### Deadlock Detection

```python
from deadlock_detection import DeadlockDetector, Process, Resource

# Create detector
detector = DeadlockDetector()

# Add processes and resources
detector.add_process(Process("P1"))
detector.add_resource(Resource("R1", capacity=1))

# Detect deadlock
is_deadlock, processes, confidence = detector.detect_deadlock_fuzzy_approach()
print(f"Deadlock detected: {is_deadlock}, Confidence: {confidence}")
```

### Graph Transformation System

```python
from graph_transformation import GraphTransformationSystem, Node, Edge, NodeType

# Create system
gts = GraphTransformationSystem()

# Add nodes and edges
gts.add_node(Node("S0", NodeType.STATE))
gts.add_edge(Edge("S0", "S1", "transition"))

# Test reachability
reachable = gts.get_reachable_states("S0")
print(f"Reachable states: {reachable}")
```

## Experimental Results

The implementation includes comprehensive experimental evaluation:

1. **Accuracy vs Problem Size**: Performance scaling analysis
2. **Execution Time Analysis**: Computational efficiency
3. **Convergence Analysis**: Algorithm convergence behavior
4. **Comparative Studies**: Comparison with baseline algorithms

## Visualization

The framework provides multiple visualization options:

- Fitness evolution plots
- Accuracy vs problem size charts
- Execution time analysis
- Deadlock detection confusion matrices
- Convergence metrics

## Methodology

This implementation follows the methodology described in Salimi et al. (2020):

1. **Fuzzy Logic Integration**: Uses triangular membership functions and fuzzy rules
2. **Genetic Algorithm**: Implements selection, crossover, and mutation operators
3. **Reachability Verification**: Tests state reachability in graph transformation systems
4. **Deadlock Detection**: Multiple detection strategies with confidence scoring
5. **Experimental Evaluation**: Comprehensive performance analysis

## Dependencies

- numpy: Numerical computations
- matplotlib: Plotting and visualization
- seaborn: Statistical visualization
- pandas: Data manipulation
- networkx: Graph algorithms
- scipy: Scientific computing
- scikit-learn: Machine learning utilities

## Citation

If you use this implementation, please cite the original paper:

```
Salimi, N., Rafe, V., Tabrizchi, H., Mosavi, A.: Fuzzy genetic algorithm approach 
for verification of reachability and detection of deadlock in graph transformation 
systems, 000241–000250 (2020)
```

## License

This implementation is provided for educational and research purposes.

## Contact

For questions or issues, please refer to the original paper or create an issue in this repository.
