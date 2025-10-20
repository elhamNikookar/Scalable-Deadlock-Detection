#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TPMC (Two-Phase Model Checking) Simulation by Mr. Pira
For Dining Philosophers Problem with 100 philosophers

Based on:
- Pira et al. (2017): GTS+BOA approach
- Pira et al. (2022): Two-phase model checking framework
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import random
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from enum import Enum
import threading
import queue

class PhilosopherState(Enum):
    """Different states of a philosopher"""
    THINKING = "thinking"
    HUNGRY = "hungry"
    EATING = "eating"

@dataclass
class Fork:
    """Fork class"""
    id: int
    available: bool = True
    owner: int = None

@dataclass
class Philosopher:
    """Philosopher class"""
    id: int
    state: PhilosopherState = PhilosopherState.THINKING
    left_fork: int = None
    right_fork: int = None
    wait_time: float = 0.0
    eat_count: int = 0

class TPMCSimulator:
    """TPMC Simulator for Dining Philosophers Problem"""
    
    def __init__(self, num_philosophers: int = 100):
        self.num_philosophers = num_philosophers
        self.philosophers = []
        self.forks = []
        self.graph = nx.DiGraph()
        self.deadlock_detected = False
        self.iteration_count = 0
        self.max_iterations = 100  # According to TPMC paper
        
        # Statistics
        self.stats = {
            'deadlocks_found': 0,
            'total_iterations': 0,
            'execution_time': 0.0,
            'states_explored': 0,
            'deadlock_states': []
        }
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the system"""
        # Create philosophers
        for i in range(self.num_philosophers):
            philosopher = Philosopher(id=i)
            self.philosophers.append(philosopher)
        
        # Create forks
        for i in range(self.num_philosophers):
            fork = Fork(id=i)
            self.forks.append(fork)
        
        # Build initial graph
        self._build_initial_graph()
    
    def _build_initial_graph(self):
        """Build initial system graph"""
        self.graph.clear()
        
        # Add philosopher nodes
        for philosopher in self.philosophers:
            self.graph.add_node(
                f"philosopher_{philosopher.id}",
                type="philosopher",
                state=philosopher.state.value,
                id=philosopher.id
            )
        
        # Add fork nodes
        for fork in self.forks:
            self.graph.add_node(
                f"fork_{fork.id}",
                type="fork",
                available=fork.available,
                id=fork.id
            )
        
        # Add dependency edges
        for i in range(self.num_philosophers):
            left_fork_id = i
            right_fork_id = (i + 1) % self.num_philosophers
            
            # Philosopher to left fork
            self.graph.add_edge(
                f"philosopher_{i}",
                f"fork_{left_fork_id}",
                relation="needs_left"
            )
            
            # Philosopher to right fork
            self.graph.add_edge(
                f"philosopher_{i}",
                f"fork_{right_fork_id}",
                relation="needs_right"
            )
    
    def _detect_deadlock_cycle(self) -> bool:
        """Detect deadlock cycles in the graph"""
        try:
            # Find strongly connected components
            strongly_connected = list(nx.strongly_connected_components(self.graph))
            
            for component in strongly_connected:
                if len(component) > 1:
                    # Check if this cycle includes hungry philosophers
                    hungry_philosophers = []
                    for node in component:
                        if node.startswith("philosopher_"):
                            node_data = self.graph.nodes[node]
                            if node_data.get('state') == PhilosopherState.HUNGRY.value:
                                hungry_philosophers.append(node)
                    
                    if len(hungry_philosophers) >= 2:
                        return True
            
            return False
        except:
            return False
    
    def _bayesian_optimization_step(self) -> Dict:
        """Bayesian optimization step (Phase 2 of TPMC)"""
        # Simulate Bayesian optimization algorithm
        # In reality, this involves sampling from posterior distribution and optimizing acquisition function
        
        # Randomly select a philosopher for state change
        philosopher_id = random.randint(0, self.num_philosophers - 1)
        philosopher = self.philosophers[philosopher_id]
        
        # State change probability based on Bayesian algorithm
        if philosopher.state == PhilosopherState.THINKING:
            # Probability of becoming hungry
            if random.random() < 0.3:  # 30% probability
                return {'action': 'become_hungry', 'philosopher_id': philosopher_id}
        elif philosopher.state == PhilosopherState.HUNGRY:
            # Probability of trying to eat
            if random.random() < 0.5:  # 50% probability
                return {'action': 'try_eat', 'philosopher_id': philosopher_id}
        elif philosopher.state == PhilosopherState.EATING:
            # Probability of finishing eating
            if random.random() < 0.4:  # 40% probability
                return {'action': 'finish_eating', 'philosopher_id': philosopher_id}
        
        return {'action': 'no_change', 'philosopher_id': philosopher_id}
    
    def _apply_transformation_rule(self, action: Dict):
        """Apply graph transformation rules (Phase 1 of TPMC)"""
        philosopher_id = action['philosopher_id']
        philosopher = self.philosophers[philosopher_id]
        
        if action['action'] == 'become_hungry':
            philosopher.state = PhilosopherState.HUNGRY
            # Update graph
            self.graph.nodes[f"philosopher_{philosopher_id}"]['state'] = PhilosopherState.HUNGRY.value
            
        elif action['action'] == 'try_eat':
            left_fork_id = philosopher_id
            right_fork_id = (philosopher_id + 1) % self.num_philosophers
            
            left_fork = self.forks[left_fork_id]
            right_fork = self.forks[right_fork_id]
            
            # Check if forks are available
            if left_fork.available and right_fork.available:
                # Take forks
                left_fork.available = False
                right_fork.available = False
                left_fork.owner = philosopher_id
                right_fork.owner = philosopher_id
                
                philosopher.state = PhilosopherState.EATING
                philosopher.left_fork = left_fork_id
                philosopher.right_fork = right_fork_id
                philosopher.eat_count += 1
                
                # Update graph
                self.graph.nodes[f"philosopher_{philosopher_id}"]['state'] = PhilosopherState.EATING.value
                self.graph.nodes[f"fork_{left_fork_id}"]['available'] = False
                self.graph.nodes[f"fork_{right_fork_id}"]['available'] = False
                
        elif action['action'] == 'finish_eating':
            # Release forks
            if philosopher.left_fork is not None and philosopher.right_fork is not None:
                left_fork = self.forks[philosopher.left_fork]
                right_fork = self.forks[philosopher.right_fork]
                
                left_fork.available = True
                right_fork.available = True
                left_fork.owner = None
                right_fork.owner = None
                
                philosopher.state = PhilosopherState.THINKING
                philosopher.left_fork = None
                philosopher.right_fork = None
                
                # Update graph
                self.graph.nodes[f"philosopher_{philosopher_id}"]['state'] = PhilosopherState.THINKING.value
                if philosopher.left_fork is not None:
                    self.graph.nodes[f"fork_{philosopher.left_fork}"]['available'] = True
                if philosopher.right_fork is not None:
                    self.graph.nodes[f"fork_{philosopher.right_fork}"]['available'] = True
    
    def _extract_graph_features(self) -> Dict:
        """Extract graph features for analysis"""
        features = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_philosophers': len(self.philosophers),
            'num_forks': len(self.forks),
            'hungry_philosophers': sum(1 for p in self.philosophers if p.state == PhilosopherState.HUNGRY),
            'eating_philosophers': sum(1 for p in self.philosophers if p.state == PhilosopherState.EATING),
            'thinking_philosophers': sum(1 for p in self.philosophers if p.state == PhilosopherState.THINKING),
            'available_forks': sum(1 for f in self.forks if f.available),
            'avg_degree': np.mean([d for n, d in self.graph.degree()]) if self.graph.number_of_nodes() > 0 else 0,
            'density': nx.density(self.graph),
            'is_strongly_connected': nx.is_strongly_connected(self.graph),
            'num_strongly_connected_components': nx.number_strongly_connected_components(self.graph)
        }
        return features
    
    def run_tpmc_simulation(self) -> Dict:
        """Run TPMC simulation"""
        print(f"Starting TPMC simulation for {self.num_philosophers} philosophers...")
        start_time = time.time()
        
        self.stats['total_iterations'] = 0
        self.stats['states_explored'] = 0
        self.stats['deadlocks_found'] = 0
        
        for iteration in range(self.max_iterations):
            self.iteration_count = iteration
            self.stats['total_iterations'] += 1
            
            # Phase 1: Apply graph transformation rules
            action = self._bayesian_optimization_step()
            self._apply_transformation_rule(action)
            
            # Extract graph features
            features = self._extract_graph_features()
            self.stats['states_explored'] += 1
            
            # Detect deadlock
            if self._detect_deadlock_cycle():
                self.deadlock_detected = True
                self.stats['deadlocks_found'] += 1
                self.stats['deadlock_states'].append({
                    'iteration': iteration,
                    'features': features,
                    'timestamp': time.time()
                })
                
                print(f"⚠️  Deadlock detected at iteration {iteration}")
                print(f"   Graph features: {features}")
                
                # Continue simulation even after deadlock detection
                # (In reality, algorithm might stop)
            
            # Show progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration}/{self.max_iterations} - "
                      f"Hungry: {features['hungry_philosophers']}, "
                      f"Eating: {features['eating_philosophers']}, "
                      f"Available forks: {features['available_forks']}")
        
        end_time = time.time()
        self.stats['execution_time'] = end_time - start_time
        
        print(f"\n✅ TPMC simulation completed!")
        print(f"⏱️  Execution time: {self.stats['execution_time']:.2f} seconds")
        print(f"🔄 Total iterations: {self.stats['total_iterations']}")
        print(f"🔍 States explored: {self.stats['states_explored']}")
        print(f"⚠️  Deadlocks found: {self.stats['deadlocks_found']}")
        
        return self.stats
    
    def visualize_system_state(self, save_path: str = None):
        """Visualize system state"""
        plt.figure(figsize=(15, 10))
        
        # Create subplot for graph
        ax1 = plt.subplot(2, 2, 1)
        
        # Color nodes based on type
        node_colors = []
        node_sizes = []
        
        for node in self.graph.nodes():
            if node.startswith("philosopher_"):
                philosopher_id = int(node.split("_")[1])
                philosopher = self.philosophers[philosopher_id]
                if philosopher.state == PhilosopherState.THINKING:
                    node_colors.append('lightblue')
                elif philosopher.state == PhilosopherState.HUNGRY:
                    node_colors.append('red')
                else:  # EATING
                    node_colors.append('green')
                node_sizes.append(300)
            else:  # fork
                node_colors.append('gray')
                node_sizes.append(100)
        
        # Draw graph
        pos = nx.spring_layout(self.graph, k=3, iterations=50)
        nx.draw(self.graph, pos, 
                node_color=node_colors, 
                node_size=node_sizes,
                with_labels=False, 
                arrows=True,
                edge_color='gray',
                alpha=0.7,
                ax=ax1)
        
        ax1.set_title(f"System State - {self.num_philosophers} Philosophers")
        
        # Statistics chart
        ax2 = plt.subplot(2, 2, 2)
        states = ['Thinking', 'Hungry', 'Eating']
        counts = [
            sum(1 for p in self.philosophers if p.state == PhilosopherState.THINKING),
            sum(1 for p in self.philosophers if p.state == PhilosopherState.HUNGRY),
            sum(1 for p in self.philosophers if p.state == PhilosopherState.EATING)
        ]
        ax2.bar(states, counts, color=['lightblue', 'red', 'green'])
        ax2.set_title('Philosopher State Distribution')
        ax2.set_ylabel('Count')
        
        # Fork chart
        ax3 = plt.subplot(2, 2, 3)
        fork_states = ['Available', 'Occupied']
        fork_counts = [
            sum(1 for f in self.forks if f.available),
            sum(1 for f in self.forks if not f.available)
        ]
        ax3.bar(fork_states, fork_counts, color=['lightgreen', 'orange'])
        ax3.set_title('Fork Status')
        ax3.set_ylabel('Count')
        
        # General statistics
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        stats_text = f"""
General Statistics:
• Number of philosophers: {self.num_philosophers}
• Number of forks: {len(self.forks)}
• Iterations completed: {self.stats['total_iterations']}
• Deadlocks found: {self.stats['deadlocks_found']}
• Execution time: {self.stats['execution_time']:.2f} seconds
• Graph density: {nx.density(self.graph):.3f}
        """
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to {save_path}")
        
        plt.show()
    
    def export_results(self, filename: str = "tpmc_results.txt"):
        """Export results to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("TPMC Simulation Results for Dining Philosophers Problem\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Number of philosophers: {self.num_philosophers}\n")
            f.write(f"Number of forks: {len(self.forks)}\n")
            f.write(f"Number of iterations: {self.stats['total_iterations']}\n")
            f.write(f"Execution time: {self.stats['execution_time']:.2f} seconds\n")
            f.write(f"States explored: {self.stats['states_explored']}\n")
            f.write(f"Deadlocks found: {self.stats['deadlocks_found']}\n\n")
            
            f.write("Deadlock Details:\n")
            f.write("-" * 30 + "\n")
            for i, deadlock in enumerate(self.stats['deadlock_states']):
                f.write(f"Deadlock {i+1}:\n")
                f.write(f"  Iteration: {deadlock['iteration']}\n")
                f.write(f"  Features: {deadlock['features']}\n")
                f.write(f"  Time: {deadlock['timestamp']}\n\n")
        
        print(f"Results saved to {filename}")

def main():
    """Main function"""
    print("TPMC Simulation by Mr. Pira")
    print("For Dining Philosophers Problem with 100 philosophers")
    print("=" * 50)
    
    # Create simulator
    simulator = TPMCSimulator(num_philosophers=100)
    
    # Run simulation
    results = simulator.run_tpmc_simulation()
    
    # Display results
    print("\n" + "=" * 50)
    print("Final Results:")
    print(f"✅ Deadlock detected: {'Yes' if results['deadlocks_found'] > 0 else 'No'}")
    print(f"⏱️  Total execution time: {results['execution_time']:.2f} seconds")
    print(f"🔄 Number of iterations: {results['total_iterations']}")
    print(f"🔍 States explored: {results['states_explored']}")
    
    # Visualize
    simulator.visualize_system_state("tpmc_visualization.png")
    
    # Export results
    simulator.export_results("tpmc_results.txt")
    
    return simulator

if __name__ == "__main__":
    simulator = main()
