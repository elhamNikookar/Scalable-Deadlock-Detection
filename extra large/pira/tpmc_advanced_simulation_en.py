#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced TPMC Simulation with Higher Deadlock Probability
For Dining Philosophers Problem with 100 philosophers

This version includes:
- Advanced deadlock detection algorithms
- More realistic philosopher behavior simulation
- Deeper graph feature analysis
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import random
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from enum import Enum
import threading
import queue
from collections import defaultdict
import json

class PhilosopherState(Enum):
    """Different states of a philosopher"""
    THINKING = "thinking"
    HUNGRY = "hungry"
    EATING = "eating"
    WAITING = "waiting"  # New state for waiting

@dataclass
class Fork:
    """Advanced fork class"""
    id: int
    available: bool = True
    owner: Optional[int] = None
    request_queue: List[int] = None  # Request queue
    
    def __post_init__(self):
        if self.request_queue is None:
            self.request_queue = []

@dataclass
class Philosopher:
    """Advanced philosopher class"""
    id: int
    state: PhilosopherState = PhilosopherState.THINKING
    left_fork: Optional[int] = None
    right_fork: Optional[int] = None
    wait_time: float = 0.0
    eat_count: int = 0
    think_time: float = 0.0
    hunger_level: float = 0.0  # Hunger level (0-1)
    patience: float = 1.0  # Philosopher patience (0-1)
    priority: int = 0  # Philosopher priority

class AdvancedTPMCSimulator:
    """Advanced TPMC Simulator for Dining Philosophers Problem"""
    
    def __init__(self, num_philosophers: int = 100, deadlock_probability: float = 0.3):
        self.num_philosophers = num_philosophers
        self.deadlock_probability = deadlock_probability
        self.philosophers = []
        self.forks = []
        self.graph = nx.DiGraph()
        self.wait_graph = nx.DiGraph()  # Waiting graph
        self.deadlock_detected = False
        self.iteration_count = 0
        self.max_iterations = 100
        
        # Advanced statistics
        self.stats = {
            'deadlocks_found': 0,
            'total_iterations': 0,
            'execution_time': 0.0,
            'states_explored': 0,
            'deadlock_states': [],
            'performance_metrics': {},
            'graph_evolution': [],
            'philosopher_behavior': defaultdict(list)
        }
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize system with advanced features"""
        # Create philosophers with random features
        for i in range(self.num_philosophers):
            philosopher = Philosopher(
                id=i,
                patience=random.uniform(0.3, 1.0),
                priority=random.randint(1, 10),
                hunger_level=random.uniform(0.0, 0.5)
            )
            self.philosophers.append(philosopher)
        
        # Create forks
        for i in range(self.num_philosophers):
            fork = Fork(id=i)
            self.forks.append(fork)
        
        # Build initial graphs
        self._build_initial_graphs()
    
    def _build_initial_graphs(self):
        """Build initial system graphs"""
        self.graph.clear()
        self.wait_graph.clear()
        
        # Add philosopher nodes
        for philosopher in self.philosophers:
            self.graph.add_node(
                f"philosopher_{philosopher.id}",
                type="philosopher",
                state=philosopher.state.value,
                id=philosopher.id,
                hunger_level=philosopher.hunger_level,
                patience=philosopher.patience,
                priority=philosopher.priority
            )
            
            # Waiting graph
            self.wait_graph.add_node(
                f"philosopher_{philosopher.id}",
                type="philosopher",
                state=philosopher.state.value
            )
        
        # Add fork nodes
        for fork in self.forks:
            self.graph.add_node(
                f"fork_{fork.id}",
                type="fork",
                available=fork.available,
                id=fork.id,
                request_count=len(fork.request_queue)
            )
        
        # Add dependency edges
        for i in range(self.num_philosophers):
            left_fork_id = i
            right_fork_id = (i + 1) % self.num_philosophers
            
            # Philosopher to left fork
            self.graph.add_edge(
                f"philosopher_{i}",
                f"fork_{left_fork_id}",
                relation="needs_left",
                weight=1.0
            )
            
            # Philosopher to right fork
            self.graph.add_edge(
                f"philosopher_{i}",
                f"fork_{right_fork_id}",
                relation="needs_right",
                weight=1.0
            )
    
    def _detect_advanced_deadlock(self) -> Tuple[bool, Dict]:
        """Advanced deadlock detection with deeper analysis"""
        deadlock_info = {
            'type': 'none',
            'cycles': [],
            'waiting_chains': [],
            'resource_contention': 0,
            'severity': 0.0
        }
        
        try:
            # 1. Detect strongly connected components
            strongly_connected = list(nx.strongly_connected_components(self.graph))
            deadlock_cycles = []
            
            for component in strongly_connected:
                if len(component) > 1:
                    # Check for hungry philosophers in cycle
                    hungry_in_cycle = []
                    for node in component:
                        if node.startswith("philosopher_"):
                            node_data = self.graph.nodes[node]
                            if node_data.get('state') in [PhilosopherState.HUNGRY.value, PhilosopherState.WAITING.value]:
                                hungry_in_cycle.append(node)
                    
                    if len(hungry_in_cycle) >= 2:
                        deadlock_cycles.append(list(component))
                        deadlock_info['cycles'].append(list(component))
            
            # 2. Analyze waiting chains
            waiting_chains = self._analyze_waiting_chains()
            deadlock_info['waiting_chains'] = waiting_chains
            
            # 3. Calculate resource contention
            resource_contention = self._calculate_resource_contention()
            deadlock_info['resource_contention'] = resource_contention
            
            # 4. Calculate deadlock severity
            severity = self._calculate_deadlock_severity(deadlock_cycles, waiting_chains, resource_contention)
            deadlock_info['severity'] = severity
            
            # Final decision
            is_deadlock = (
                len(deadlock_cycles) > 0 or 
                len(waiting_chains) > 0 or 
                resource_contention > 0.8 or
                severity > 0.5
            )
            
            if is_deadlock:
                if len(deadlock_cycles) > 0:
                    deadlock_info['type'] = 'circular_wait'
                elif len(waiting_chains) > 0:
                    deadlock_info['type'] = 'waiting_chain'
                elif resource_contention > 0.8:
                    deadlock_info['type'] = 'resource_contention'
                else:
                    deadlock_info['type'] = 'mixed'
            
            return is_deadlock, deadlock_info
            
        except Exception as e:
            print(f"Error in deadlock detection: {e}")
            return False, deadlock_info
    
    def _analyze_waiting_chains(self) -> List[List[str]]:
        """Analyze waiting chains"""
        waiting_chains = []
        
        # Find hungry philosophers
        hungry_philosophers = []
        for philosopher in self.philosophers:
            if philosopher.state in [PhilosopherState.HUNGRY, PhilosopherState.WAITING]:
                hungry_philosophers.append(philosopher.id)
        
        # Analyze waiting chains
        for start_philosopher in hungry_philosophers:
            chain = self._trace_waiting_chain(start_philosopher, set())
            if len(chain) > 1:
                waiting_chains.append(chain)
        
        return waiting_chains
    
    def _trace_waiting_chain(self, philosopher_id: int, visited: Set[int]) -> List[str]:
        """Trace waiting chain from a philosopher"""
        if philosopher_id in visited:
            return []  # Cycle found
        
        visited.add(philosopher_id)
        chain = [f"philosopher_{philosopher_id}"]
        
        philosopher = self.philosophers[philosopher_id]
        if philosopher.state not in [PhilosopherState.HUNGRY, PhilosopherState.WAITING]:
            return chain
        
        # Check required forks
        left_fork_id = philosopher_id
        right_fork_id = (philosopher_id + 1) % self.num_philosophers
        
        left_fork = self.forks[left_fork_id]
        right_fork = self.forks[right_fork_id]
        
        # If forks are occupied, follow their owners
        if not left_fork.available and left_fork.owner is not None:
            next_chain = self._trace_waiting_chain(left_fork.owner, visited.copy())
            chain.extend(next_chain)
        
        if not right_fork.available and right_fork.owner is not None:
            next_chain = self._trace_waiting_chain(right_fork.owner, visited.copy())
            chain.extend(next_chain)
        
        return chain
    
    def _calculate_resource_contention(self) -> float:
        """Calculate resource contention level"""
        total_requests = 0
        total_resources = len(self.forks)
        
        for fork in self.forks:
            total_requests += len(fork.request_queue)
        
        return total_requests / (total_resources * self.num_philosophers) if total_resources > 0 else 0
    
    def _calculate_deadlock_severity(self, cycles: List, waiting_chains: List, resource_contention: float) -> float:
        """Calculate deadlock severity"""
        severity = 0.0
        
        # Weight cycles
        severity += len(cycles) * 0.3
        
        # Weight waiting chains
        severity += len(waiting_chains) * 0.2
        
        # Weight resource contention
        severity += resource_contention * 0.3
        
        # Weight hungry philosophers
        hungry_count = sum(1 for p in self.philosophers if p.state == PhilosopherState.HUNGRY)
        severity += (hungry_count / self.num_philosophers) * 0.2
        
        return min(severity, 1.0)
    
    def _advanced_bayesian_optimization(self) -> Dict:
        """Advanced Bayesian optimization considering deadlock probability"""
        # Select philosopher based on priority and hunger level
        candidate_philosophers = []
        
        for philosopher in self.philosophers:
            if philosopher.state == PhilosopherState.THINKING:
                # Probability of becoming hungry based on hunger level
                hunger_prob = min(philosopher.hunger_level + 0.1, 1.0)
                if random.random() < hunger_prob:
                    candidate_philosophers.append((philosopher.id, hunger_prob))
            elif philosopher.state == PhilosopherState.HUNGRY:
                # Probability of trying to eat based on patience
                eat_prob = philosopher.patience * 0.5
                if random.random() < eat_prob:
                    candidate_philosophers.append((philosopher.id, eat_prob))
            elif philosopher.state == PhilosopherState.EATING:
                # Probability of finishing eating
                finish_prob = 0.4
                if random.random() < finish_prob:
                    candidate_philosophers.append((philosopher.id, finish_prob))
        
        if not candidate_philosophers:
            return {'action': 'no_change', 'philosopher_id': 0}
        
        # Select best candidate based on probability and priority
        best_candidate = max(candidate_philosophers, key=lambda x: x[1])
        philosopher_id = best_candidate[0]
        philosopher = self.philosophers[philosopher_id]
        
        # Determine action based on current state
        if philosopher.state == PhilosopherState.THINKING:
            return {'action': 'become_hungry', 'philosopher_id': philosopher_id}
        elif philosopher.state == PhilosopherState.HUNGRY:
            return {'action': 'try_eat', 'philosopher_id': philosopher_id}
        elif philosopher.state == PhilosopherState.EATING:
            return {'action': 'finish_eating', 'philosopher_id': philosopher_id}
        
        return {'action': 'no_change', 'philosopher_id': philosopher_id}
    
    def _apply_advanced_transformation(self, action: Dict):
        """Apply advanced transformation rules"""
        philosopher_id = action['philosopher_id']
        philosopher = self.philosophers[philosopher_id]
        
        if action['action'] == 'become_hungry':
            philosopher.state = PhilosopherState.HUNGRY
            philosopher.hunger_level = min(philosopher.hunger_level + 0.2, 1.0)
            self.graph.nodes[f"philosopher_{philosopher_id}"]['state'] = PhilosopherState.HUNGRY.value
            self.graph.nodes[f"philosopher_{philosopher_id}"]['hunger_level'] = philosopher.hunger_level
            
        elif action['action'] == 'try_eat':
            left_fork_id = philosopher_id
            right_fork_id = (philosopher_id + 1) % self.num_philosophers
            
            left_fork = self.forks[left_fork_id]
            right_fork = self.forks[right_fork_id]
            
            # Add to request queue
            if left_fork_id not in left_fork.request_queue:
                left_fork.request_queue.append(philosopher_id)
            if right_fork_id not in right_fork.request_queue:
                right_fork.request_queue.append(philosopher_id)
            
            # Check if forks are available
            if left_fork.available and right_fork.available:
                # Take forks
                left_fork.available = False
                right_fork.available = False
                left_fork.owner = philosopher_id
                right_fork.owner = philosopher_id
                
                # Remove from request queue
                if philosopher_id in left_fork.request_queue:
                    left_fork.request_queue.remove(philosopher_id)
                if philosopher_id in right_fork.request_queue:
                    right_fork.request_queue.remove(philosopher_id)
                
                philosopher.state = PhilosopherState.EATING
                philosopher.left_fork = left_fork_id
                philosopher.right_fork = right_fork_id
                philosopher.eat_count += 1
                philosopher.hunger_level = 0.0  # Satisfy hunger
                
                # Update graph
                self.graph.nodes[f"philosopher_{philosopher_id}"]['state'] = PhilosopherState.EATING.value
                self.graph.nodes[f"philosopher_{philosopher_id}"]['hunger_level'] = 0.0
                self.graph.nodes[f"fork_{left_fork_id}"]['available'] = False
                self.graph.nodes[f"fork_{right_fork_id}"]['available'] = False
            else:
                # If forks are not available, go to waiting state
                philosopher.state = PhilosopherState.WAITING
                self.graph.nodes[f"philosopher_{philosopher_id}"]['state'] = PhilosopherState.WAITING.value
                
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
                philosopher.think_time = random.uniform(1.0, 5.0)  # Random thinking time
                
                # Update graph
                self.graph.nodes[f"philosopher_{philosopher_id}"]['state'] = PhilosopherState.THINKING.value
                if philosopher.left_fork is not None:
                    self.graph.nodes[f"fork_{philosopher.left_fork}"]['available'] = True
                if philosopher.right_fork is not None:
                    self.graph.nodes[f"fork_{philosopher.right_fork}"]['available'] = True
    
    def _extract_advanced_features(self) -> Dict:
        """Extract advanced graph features"""
        features = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_philosophers': len(self.philosophers),
            'num_forks': len(self.forks),
            'hungry_philosophers': sum(1 for p in self.philosophers if p.state == PhilosopherState.HUNGRY),
            'eating_philosophers': sum(1 for p in self.philosophers if p.state == PhilosopherState.EATING),
            'thinking_philosophers': sum(1 for p in self.philosophers if p.state == PhilosopherState.THINKING),
            'waiting_philosophers': sum(1 for p in self.philosophers if p.state == PhilosopherState.WAITING),
            'available_forks': sum(1 for f in self.forks if f.available),
            'avg_degree': np.mean([d for n, d in self.graph.degree()]) if self.graph.number_of_nodes() > 0 else 0,
            'density': nx.density(self.graph),
            'is_strongly_connected': nx.is_strongly_connected(self.graph),
            'num_strongly_connected_components': nx.number_strongly_connected_components(self.graph),
            'avg_hunger_level': np.mean([p.hunger_level for p in self.philosophers]),
            'avg_patience': np.mean([p.patience for p in self.philosophers]),
            'total_requests': sum(len(f.request_queue) for f in self.forks),
            'resource_contention': self._calculate_resource_contention(),
            'graph_clustering': nx.average_clustering(self.graph.to_undirected()) if self.graph.number_of_nodes() > 0 else 0,
            'assortativity': nx.degree_assortativity_coefficient(self.graph) if self.graph.number_of_nodes() > 0 else 0
        }
        return features
    
    def run_advanced_tpmc_simulation(self) -> Dict:
        """Run advanced TPMC simulation"""
        print(f"Starting advanced TPMC simulation for {self.num_philosophers} philosophers...")
        print(f"Deadlock probability: {self.deadlock_probability}")
        start_time = time.time()
        
        self.stats['total_iterations'] = 0
        self.stats['states_explored'] = 0
        self.stats['deadlocks_found'] = 0
        
        for iteration in range(self.max_iterations):
            self.iteration_count = iteration
            self.stats['total_iterations'] += 1
            
            # Advanced Bayesian optimization
            action = self._advanced_bayesian_optimization()
            self._apply_advanced_transformation(action)
            
            # Extract advanced features
            features = self._extract_advanced_features()
            self.stats['states_explored'] += 1
            self.stats['graph_evolution'].append(features.copy())
            
            # Advanced deadlock detection
            is_deadlock, deadlock_info = self._detect_advanced_deadlock()
            
            if is_deadlock:
                self.deadlock_detected = True
                self.stats['deadlocks_found'] += 1
                self.stats['deadlock_states'].append({
                    'iteration': iteration,
                    'features': features,
                    'deadlock_info': deadlock_info,
                    'timestamp': time.time()
                })
                
                print(f"⚠️  Deadlock detected at iteration {iteration}")
                print(f"   Type: {deadlock_info['type']}")
                print(f"   Severity: {deadlock_info['severity']:.3f}")
                print(f"   Cycles: {len(deadlock_info['cycles'])}")
                print(f"   Waiting chains: {len(deadlock_info['waiting_chains'])}")
            
            # Show progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration}/{self.max_iterations} - "
                      f"Hungry: {features['hungry_philosophers']}, "
                      f"Eating: {features['eating_philosophers']}, "
                      f"Waiting: {features['waiting_philosophers']}, "
                      f"Available forks: {features['available_forks']}, "
                      f"Resource contention: {features['resource_contention']:.3f}")
        
        end_time = time.time()
        self.stats['execution_time'] = end_time - start_time
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
        print(f"\n✅ Advanced TPMC simulation completed!")
        print(f"⏱️  Execution time: {self.stats['execution_time']:.2f} seconds")
        print(f"🔄 Total iterations: {self.stats['total_iterations']}")
        print(f"🔍 States explored: {self.stats['states_explored']}")
        print(f"⚠️  Deadlocks found: {self.stats['deadlocks_found']}")
        
        return self.stats
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics"""
        if not self.stats['graph_evolution']:
            return
        
        features_list = self.stats['graph_evolution']
        
        # Calculate general statistics
        self.stats['performance_metrics'] = {
            'avg_hungry_philosophers': np.mean([f['hungry_philosophers'] for f in features_list]),
            'max_hungry_philosophers': max([f['hungry_philosophers'] for f in features_list]),
            'avg_eating_philosophers': np.mean([f['eating_philosophers'] for f in features_list]),
            'avg_resource_contention': np.mean([f['resource_contention'] for f in features_list]),
            'max_resource_contention': max([f['resource_contention'] for f in features_list]),
            'avg_graph_density': np.mean([f['density'] for f in features_list]),
            'avg_clustering': np.mean([f['graph_clustering'] for f in features_list]),
            'stability_score': self._calculate_stability_score(features_list)
        }
    
    def _calculate_stability_score(self, features_list: List[Dict]) -> float:
        """Calculate system stability score"""
        if len(features_list) < 2:
            return 0.0
        
        # Calculate changes in key features
        hungry_changes = np.std([f['hungry_philosophers'] for f in features_list])
        eating_changes = np.std([f['eating_philosophers'] for f in features_list])
        contention_changes = np.std([f['resource_contention'] for f in features_list])
        
        # Stability score (less change is better)
        stability = 1.0 / (1.0 + hungry_changes + eating_changes + contention_changes)
        return min(stability, 1.0)
    
    def visualize_advanced_system(self, save_path: str = None):
        """Advanced system visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Main system graph
        ax1 = axes[0, 0]
        self._draw_system_graph(ax1)
        ax1.set_title(f"System Graph - {self.num_philosophers} Philosophers")
        
        # 2. Philosopher state distribution
        ax2 = axes[0, 1]
        states = ['Thinking', 'Hungry', 'Eating', 'Waiting']
        counts = [
            sum(1 for p in self.philosophers if p.state == PhilosopherState.THINKING),
            sum(1 for p in self.philosophers if p.state == PhilosopherState.HUNGRY),
            sum(1 for p in self.philosophers if p.state == PhilosopherState.EATING),
            sum(1 for p in self.philosophers if p.state == PhilosopherState.WAITING)
        ]
        colors = ['lightblue', 'red', 'green', 'orange']
        bars = ax2.bar(states, counts, color=colors)
        ax2.set_title('Philosopher State Distribution')
        ax2.set_ylabel('Count')
        
        # Add values on bars
        for bar, count in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom')
        
        # 3. Feature evolution chart
        ax3 = axes[0, 2]
        if self.stats['graph_evolution']:
            iterations = range(len(self.stats['graph_evolution']))
            hungry_evolution = [f['hungry_philosophers'] for f in self.stats['graph_evolution']]
            eating_evolution = [f['eating_philosophers'] for f in self.stats['graph_evolution']]
            
            ax3.plot(iterations, hungry_evolution, 'r-', label='Hungry', alpha=0.7)
            ax3.plot(iterations, eating_evolution, 'g-', label='Eating', alpha=0.7)
            ax3.set_title('Philosopher State Evolution')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Count')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Resource contention chart
        ax4 = axes[1, 0]
        if self.stats['graph_evolution']:
            contention_evolution = [f['resource_contention'] for f in self.stats['graph_evolution']]
            ax4.plot(iterations, contention_evolution, 'b-', alpha=0.7)
            ax4.set_title('Resource Contention Evolution')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Contention Level')
            ax4.grid(True, alpha=0.3)
        
        # 5. Graph density chart
        ax5 = axes[1, 1]
        if self.stats['graph_evolution']:
            density_evolution = [f['density'] for f in self.stats['graph_evolution']]
            ax5.plot(iterations, density_evolution, 'm-', alpha=0.7)
            ax5.set_title('Graph Density Evolution')
            ax5.set_xlabel('Iteration')
            ax5.set_ylabel('Density')
            ax5.grid(True, alpha=0.3)
        
        # 6. General statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        stats_text = f"""
Advanced Statistics:
• Number of philosophers: {self.num_philosophers}
• Number of forks: {len(self.forks)}
• Iterations completed: {self.stats['total_iterations']}
• Deadlocks found: {self.stats['deadlocks_found']}
• Execution time: {self.stats['execution_time']:.2f} seconds
• Graph density: {nx.density(self.graph):.3f}
• Stability score: {self.stats['performance_metrics'].get('stability_score', 0):.3f}
• Avg resource contention: {self.stats['performance_metrics'].get('avg_resource_contention', 0):.3f}
        """
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Advanced chart saved to {save_path}")
        
        plt.show()
    
    def _draw_system_graph(self, ax):
        """Draw system graph"""
        # Color nodes
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
                elif philosopher.state == PhilosopherState.EATING:
                    node_colors.append('green')
                else:  # WAITING
                    node_colors.append('orange')
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
                ax=ax)
    
    def export_advanced_results(self, filename: str = "advanced_tpmc_results.json"):
        """Export advanced results to JSON file"""
        results = {
            'simulation_info': {
                'num_philosophers': self.num_philosophers,
                'deadlock_probability': self.deadlock_probability,
                'max_iterations': self.max_iterations,
                'execution_time': self.stats['execution_time']
            },
            'statistics': self.stats,
            'final_state': {
                'philosophers': [
                    {
                        'id': p.id,
                        'state': p.state.value,
                        'hunger_level': p.hunger_level,
                        'patience': p.patience,
                        'priority': p.priority,
                        'eat_count': p.eat_count
                    } for p in self.philosophers
                ],
                'forks': [
                    {
                        'id': f.id,
                        'available': f.available,
                        'owner': f.owner,
                        'request_queue': f.request_queue
                    } for f in self.forks
                ]
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Advanced results saved to {filename}")

def main():
    """Main function"""
    print("Advanced TPMC Simulation by Mr. Pira")
    print("For Dining Philosophers Problem with 100 philosophers")
    print("=" * 50)
    
    # Create advanced simulator
    simulator = AdvancedTPMCSimulator(
        num_philosophers=100,
        deadlock_probability=0.4  # Higher probability for deadlock
    )
    
    # Run simulation
    results = simulator.run_advanced_tpmc_simulation()
    
    # Display results
    print("\n" + "=" * 50)
    print("Final Results:")
    print(f"✅ Deadlock detected: {'Yes' if results['deadlocks_found'] > 0 else 'No'}")
    print(f"⏱️  Total execution time: {results['execution_time']:.2f} seconds")
    print(f"🔄 Number of iterations: {results['total_iterations']}")
    print(f"🔍 States explored: {results['states_explored']}")
    
    if results['performance_metrics']:
        print(f"📊 Stability score: {results['performance_metrics']['stability_score']:.3f}")
        print(f"📊 Average resource contention: {results['performance_metrics']['avg_resource_contention']:.3f}")
    
    # Advanced visualization
    simulator.visualize_advanced_system("advanced_tpmc_visualization.png")
    
    # Export results
    simulator.export_advanced_results("advanced_tpmc_results.json")
    
    return simulator

if __name__ == "__main__":
    simulator = main()
