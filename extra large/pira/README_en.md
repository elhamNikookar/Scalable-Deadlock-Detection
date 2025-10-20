# TPMC Simulation by Mr. Pira for Dining Philosophers Problem

This project includes implementation and simulation of the TPMC (Two-Phase Model Checking) approach by Mr. Pira for the Dining Philosophers Problem with 100 philosophers.

## 📋 Table of Contents

- [Introduction](#introduction)
- [TPMC Approach](#tpmc-approach)
- [Project Files](#project-files)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Results](#results)
- [Features](#features)

## 🔍 Introduction

The Dining Philosophers Problem is one of the classic problems in computer science used to understand concurrency and deadlock concepts. In this problem, several philosophers sit around a table, and each needs two forks (one on the left and one on the right) to eat.

### The Problem:
- Each philosopher is either thinking, hungry, or eating
- To eat, each philosopher needs two forks
- Forks are shared (each fork is shared between two philosophers)
- If all philosophers pick up their left fork simultaneously, a deadlock occurs

## 🔬 TPMC Approach

TPMC (Two-Phase Model Checking) is an approach developed by Mr. Pira and colleagues that includes two main phases:

### Phase 1: Graph Transformation Systems (GTS)
- System modeling as graphs
- Definition of transformation rules for state changes
- Creation of system state space

### Phase 2: Bayesian Optimization
- Use of Bayesian optimization algorithms
- Intelligent search in state space
- High-accuracy deadlock detection

### Key TPMC Features:
- **100 iterations** for the algorithm
- **Exploration bounds** proportional to problem size
- **Advanced deadlock detection** with cycle and waiting chain analysis
- **Bayesian optimization** for selecting the best action

## 📁 Project Files

### 1. `tpmc_dining_philosophers_en.py`
Basic TPMC simulator including:
- Dining Philosophers Problem implementation
- Simple TPMC algorithm
- Deadlock detection with strongly connected component analysis
- Results visualization

### 2. `tpmc_advanced_simulation_en.py`
Advanced TPMC simulator including:
- Advanced deadlock detection (cycles, waiting chains, resource contention)
- More realistic philosopher behavior simulation
- Deeper graph feature analysis
- Advanced statistics and performance metrics
- JSON results export

### 3. `main.tex`, `main2.tex`, `main_2.tex`
LaTeX files containing papers and research related to TPMC and deadlock detection methods

## 🚀 Installation & Setup

### Prerequisites:
```bash
pip install numpy networkx matplotlib
```

### Running the code:
```bash
# Basic version
python tpmc_dining_philosophers_en.py

# Advanced version
python tpmc_advanced_simulation_en.py
```

## 💻 Usage

### Basic Version:
```python
from tpmc_dining_philosophers_en import TPMCSimulator

# Create simulator for 100 philosophers
simulator = TPMCSimulator(num_philosophers=100)

# Run simulation
results = simulator.run_tpmc_simulation()

# Display results
simulator.visualize_system_state("results.png")
simulator.export_results("results.txt")
```

### Advanced Version:
```python
from tpmc_advanced_simulation_en import AdvancedTPMCSimulator

# Create advanced simulator
simulator = AdvancedTPMCSimulator(
    num_philosophers=100,
    deadlock_probability=0.4
)

# Run simulation
results = simulator.run_advanced_tpmc_simulation()

# Display advanced results
simulator.visualize_advanced_system("advanced_results.png")
simulator.export_advanced_results("advanced_results.json")
```

## 📊 Results

### Basic Version Results:
- **Execution time**: ~0.07 seconds
- **Number of iterations**: 100
- **States explored**: 100
- **Deadlocks found**: 0 (in normal execution)

### Advanced Version Results:
- **Execution time**: ~0.40 seconds
- **Number of iterations**: 100
- **States explored**: 100
- **Deadlocks found**: 36 (with deadlock probability 0.4)
- **Stability score**: 0.074
- **Average resource contention**: 0.000

## ✨ Features

### Basic Version Features:
- ✅ Dining Philosophers Problem implementation
- ✅ TPMC algorithm with 100 iterations
- ✅ Deadlock detection with cycle analysis
- ✅ System visualization
- ✅ Text results export

### Advanced Version Features:
- ✅ Advanced deadlock detection (cycles, waiting chains, resource contention)
- ✅ More realistic simulation with additional states (waiting)
- ✅ Deeper graph feature analysis
- ✅ Advanced statistics and performance metrics
- ✅ JSON results export
- ✅ System evolution charts
- ✅ Stability score calculation

## 🔧 Configuration

### Adjustable Parameters:

#### Basic Version:
- `num_philosophers`: Number of philosophers (default: 100)
- `max_iterations`: Maximum number of iterations (default: 100)

#### Advanced Version:
- `num_philosophers`: Number of philosophers (default: 100)
- `deadlock_probability`: Deadlock probability (default: 0.4)
- `max_iterations`: Maximum number of iterations (default: 100)

## 📈 Results Analysis

### Performance Metrics:
1. **Stability Score**: Higher values indicate more stable system
2. **Resource Contention**: Lower values indicate less resource competition
3. **Graph Density**: Indicates complexity of relationships in the system
4. **Number of Deadlocks**: Indicates algorithm efficiency

### Types of Detected Deadlocks:
1. **Circular Wait**: Circular waiting patterns
2. **Waiting Chain**: Linear waiting chains
3. **Resource Contention**: Intense resource competition
4. **Mixed**: Combination of the above types

## 🎯 Applications

This simulator can be used for:
- Teaching concurrency and deadlock concepts
- Testing deadlock detection algorithms
- Research in distributed systems
- Analysis of complex system behavior
- Development of new deadlock detection methods

## 📚 References

1. Pira et al. (2017): "GTS+BOA approach for deadlock detection"
2. Pira et al. (2022): "Two-phase model checking framework"
3. Pira et al. (2016): "GTS+DataMining for efficient model checking"
4. Pira et al. (2022): "GTS+Neural Networks for AI planning"

## 🤝 Contributing

To contribute to this project:
1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Submit a pull request

## 📄 License

This project is released under the MIT License.

## 👥 Authors

- **Paper Analysis**: Based on Mr. Pira and colleagues' research
- **Implementation**: TPMC simulator for Dining Philosophers Problem
- **Development**: Basic and advanced versions

---

**Note**: This simulator is designed for educational and research purposes and may require additional configuration for use in production environments.
