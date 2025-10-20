#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified SDD (Scalable Deadlock Detection) Approach
Using Graph Transformation Systems without deep learning dependencies

This is a simplified version that focuses on the core SDD methodology
without requiring PyTorch or other heavy dependencies.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import random
from typing import List, Tuple, Dict, Set, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
from collections import defaultdict, deque

class SystemState(Enum):
    """System states for deadlock detection"""
    NORMAL = "normal"
    SUSPICIOUS = "suspicious"
    DEADLOCK = "deadlock"
    RECOVERY = "recovery"

@dataclass
class Process:
    """Process representation in the system"""
    id: int
    state: str = "running"
    resources: Set[int] = None
    waiting_for: Set[int] = None
    priority: int = 1
    execution_time: float = 0.0
    deadlock_risk: float = 0.0
    
    def __post_init__(self):
        if self.resources is None:
            self.resources = set()
        if self.waiting_for is None:
            self.waiting_for = set()

@dataclass
class Resource:
    """Resource representation in the system"""
    id: int
    available: bool = True
    owner: Optional[int] = None
    request_queue: List[int] = None
    type: str = "shared"
    contention_level: float = 0.0
    
    def __post_init__(self):
        if self.request_queue is None:
            self.request_queue = []

class GraphTransformationSystem:
    """Graph Transformation System for modeling system evolution"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.transformation_rules = []
        self.state_history = []
        self.deadlock_patterns = []
        
    def add_node(self, node_id: str, node_type: str, **attributes):
        """Add a node to the graph"""
        self.graph.add_node(node_id, type=node_type, **attributes)
    
    def add_edge(self, source: str, target: str, edge_type: str, **attributes):
        """Add an edge to the graph"""
        self.graph.add_edge(source, target, type=edge_type, **attributes)
    
    def define_transformation_rule(self, name: str, pattern: Dict, replacement: Dict, condition: callable = None):
        """Define a transformation rule"""
        rule = {
            'name': name,
            'pattern': pattern,
            'replacement': replacement,
            'condition': condition
        }
        self.transformation_rules.append(rule)
    
    def apply_transformation(self, rule_name: str, context: Dict) -> bool:
        """Apply a transformation rule"""
        for rule in self.transformation_rules:
            if rule['name'] == rule_name:
                if rule['condition'] is None or rule['condition'](context):
                    return self._execute_transformation(rule, context)
        return False
    
    def _execute_transformation(self, rule: Dict, context: Dict) -> bool:
        """Execute a transformation rule"""
        try:
            if rule['name'] == 'process_request_resource':
                process_id = context.get('process_id')
                resource_id = context.get('resource_id')
                if process_id and resource_id:
                    self.graph.add_edge(f"process_{process_id}", f"resource_{resource_id}", 
                                      type="requests", weight=1.0)
                    return True
            elif rule['name'] == 'process_acquire_resource':
                process_id = context.get('process_id')
                resource_id = context.get('resource_id')
                if process_id and resource_id:
                    self.graph.add_edge(f"process_{process_id}", f"resource_{resource_id}", 
                                      type="owns", weight=1.0)
                    return True
            elif rule['name'] == 'process_release_resource':
                process_id = context.get('process_id')
                resource_id = context.get('resource_id')
                if process_id and resource_id:
                    if self.graph.has_edge(f"process_{process_id}", f"resource_{resource_id}"):
                        self.graph.remove_edge(f"process_{process_id}", f"resource_{resource_id}")
                    return True
            return False
        except Exception as e:
            print(f"Error executing transformation {rule['name']}: {e}")
            return False
    
    def get_graph_features(self) -> Dict:
        """Extract features from the current graph state"""
        features = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'avg_degree': np.mean([d for n, d in self.graph.degree()]) if self.graph.number_of_nodes() > 0 else 0,
            'is_strongly_connected': nx.is_strongly_connected(self.graph),
            'num_strongly_connected_components': nx.number_strongly_connected_components(self.graph),
            'clustering_coefficient': nx.average_clustering(self.graph.to_undirected()) if self.graph.number_of_nodes() > 0 else 0,
            'assortativity': nx.degree_assortativity_coefficient(self.graph) if self.graph.number_of_nodes() > 0 else 0,
            'avg_path_length': nx.average_shortest_path_length(self.graph) if nx.is_connected(self.graph.to_undirected()) else float('inf')
        }
        return features

class SimplifiedGNN:
    """Simplified Graph Neural Network using traditional ML approaches"""
    
    def __init__(self):
        self.feature_weights = np.random.randn(10)  # Random weights for features
        self.bias = 0.0
        self.learning_rate = 0.01
        
    def extract_features(self, graph: nx.DiGraph, processes: List[Process], resources: List[Resource]) -> np.ndarray:
        """Extract features for deadlock prediction"""
        features = []
        
        # Graph-level features
        features.append(graph.number_of_nodes() / 100.0)  # Normalized node count
        features.append(graph.number_of_edges() / 100.0)  # Normalized edge count
        features.append(nx.density(graph))  # Graph density
        features.append(nx.number_strongly_connected_components(graph) / 10.0)  # SCC count
        
        # Process-level features
        running_processes = sum(1 for p in processes if p.state == 'running')
        waiting_processes = sum(1 for p in processes if p.state == 'waiting')
        features.append(running_processes / len(processes))  # Running ratio
        features.append(waiting_processes / len(processes))  # Waiting ratio
        
        # Resource-level features
        available_resources = sum(1 for r in resources if r.available)
        features.append(available_resources / len(resources))  # Availability ratio
        
        # Contention features
        total_requests = sum(len(r.request_queue) for r in resources)
        features.append(total_requests / len(resources))  # Average requests per resource
        
        # Deadlock risk features
        avg_deadlock_risk = np.mean([p.deadlock_risk for p in processes])
        features.append(avg_deadlock_risk)  # Average deadlock risk
        
        return np.array(features)
    
    def predict_deadlock(self, features: np.ndarray) -> Tuple[bool, float]:
        """Predict deadlock using simplified neural network"""
        # Simple linear combination
        score = np.dot(features, self.feature_weights) + self.bias
        
        # Sigmoid activation
        probability = 1 / (1 + np.exp(-score))
        
        is_deadlock = probability > 0.5
        return is_deadlock, probability
    
    def update_weights(self, features: np.ndarray, actual_deadlock: bool, predicted_deadlock: bool):
        """Update weights based on prediction accuracy"""
        if actual_deadlock != predicted_deadlock:
            # Update weights
            error = 1.0 if actual_deadlock else -1.0
            self.feature_weights += self.learning_rate * error * features
            self.bias += self.learning_rate * error

class SDDDetector:
    """Simplified SDD Detector using GTS and simplified ML"""
    
    def __init__(self, num_processes: int = 100, num_resources: int = 50):
        self.num_processes = num_processes
        self.num_resources = num_resources
        self.gts = GraphTransformationSystem()
        self.gnn = SimplifiedGNN()
        self.processes = []
        self.resources = []
        self.deadlock_history = []
        self.performance_metrics = {}
        
        # Initialize system
        self._initialize_system()
        self._setup_transformation_rules()
    
    def _initialize_system(self):
        """Initialize the system with processes and resources"""
        # Create processes
        for i in range(self.num_processes):
            process = Process(
                id=i,
                state="running",
                priority=random.randint(1, 10),
                execution_time=random.uniform(0.1, 5.0),
                deadlock_risk=random.uniform(0.0, 0.3)
            )
            self.processes.append(process)
            self.gts.add_node(f"process_{i}", "process", 
                            state=process.state, 
                            priority=process.priority,
                            deadlock_risk=process.deadlock_risk)
        
        # Create resources
        for i in range(self.num_resources):
            resource = Resource(
                id=i,
                type="shared" if random.random() < 0.7 else "exclusive",
                contention_level=random.uniform(0.0, 0.5)
            )
            self.resources.append(resource)
            self.gts.add_node(f"resource_{i}", "resource", 
                            available=resource.available,
                            type=resource.type,
                            contention_level=resource.contention_level)
    
    def _setup_transformation_rules(self):
        """Setup transformation rules for the system"""
        # Process requests resource
        self.gts.define_transformation_rule(
            'process_request_resource',
            {'type': 'process'}, {'type': 'process'},
            lambda ctx: True
        )
        
        # Process acquires resource
        self.gts.define_transformation_rule(
            'process_acquire_resource',
            {'type': 'process'}, {'type': 'process'},
            lambda ctx: True
        )
        
        # Process releases resource
        self.gts.define_transformation_rule(
            'process_release_resource',
            {'type': 'process'}, {'type': 'process'},
            lambda ctx: True
        )
    
    def _detect_deadlock_traditional(self) -> Tuple[bool, Dict]:
        """Traditional deadlock detection using cycle detection"""
        try:
            # Find strongly connected components
            sccs = list(nx.strongly_connected_components(self.gts.graph))
            deadlock_info = {
                'type': 'none',
                'cycles': [],
                'severity': 0.0,
                'affected_processes': []
            }
            
            for scc in sccs:
                if len(scc) > 1:
                    # Check if this is a deadlock cycle
                    process_nodes = [n for n in scc if n.startswith('process_')]
                    if len(process_nodes) >= 2:
                        deadlock_info['cycles'].append(list(scc))
                        deadlock_info['affected_processes'].extend(process_nodes)
            
            is_deadlock = len(deadlock_info['cycles']) > 0
            if is_deadlock:
                deadlock_info['type'] = 'circular_wait'
                deadlock_info['severity'] = len(deadlock_info['cycles']) / self.num_processes
            
            return is_deadlock, deadlock_info
            
        except Exception as e:
            print(f"Error in traditional deadlock detection: {e}")
            return False, {'type': 'none', 'cycles': [], 'severity': 0.0, 'affected_processes': []}
    
    def _detect_deadlock_ml(self) -> Tuple[bool, float]:
        """Deadlock detection using simplified ML approach"""
        try:
            # Extract features
            features = self.gnn.extract_features(self.gts.graph, self.processes, self.resources)
            
            # Make prediction
            is_deadlock, probability = self.gnn.predict_deadlock(features)
            
            return is_deadlock, probability
            
        except Exception as e:
            print(f"Error in ML deadlock detection: {e}")
            return False, 0.0
    
    def _simulate_system_evolution(self, num_steps: int = 100):
        """Simulate system evolution using transformation rules"""
        for step in range(num_steps):
            # Randomly select a process
            process_id = random.randint(0, self.num_processes - 1)
            process = self.processes[process_id]
            
            # Randomly select an action
            action = random.choice(['request', 'acquire', 'release', 'idle'])
            
            if action == 'request' and process.state == 'running':
                # Request a random resource
                resource_id = random.randint(0, self.num_resources - 1)
                resource = self.resources[resource_id]
                
                if resource.available:
                    # Add to request queue
                    if process_id not in resource.request_queue:
                        resource.request_queue.append(process_id)
                    
                    # Apply transformation
                    self.gts.apply_transformation('process_request_resource', {
                        'process_id': process_id,
                        'resource_id': resource_id
                    })
            
            elif action == 'acquire' and process.state == 'running':
                # Try to acquire a resource
                resource_id = random.randint(0, self.num_resources - 1)
                resource = self.resources[resource_id]
                
                if (resource.available and 
                    process_id in resource.request_queue and 
                    resource.request_queue.index(process_id) == 0):
                    
                    # Acquire resource
                    resource.available = False
                    resource.owner = process_id
                    resource.request_queue.remove(process_id)
                    process.resources.add(resource_id)
                    
                    # Apply transformation
                    self.gts.apply_transformation('process_acquire_resource', {
                        'process_id': process_id,
                        'resource_id': resource_id
                    })
            
            elif action == 'release' and process.resources:
                # Release a random resource
                resource_id = random.choice(list(process.resources))
                resource = self.resources[resource_id]
                
                # Release resource
                resource.available = True
                resource.owner = None
                process.resources.remove(resource_id)
                
                # Apply transformation
                self.gts.apply_transformation('process_release_resource', {
                    'process_id': process_id,
                    'resource_id': resource_id
                })
            
            # Update process state
            if not process.resources and process.state == 'running':
                process.state = 'waiting'
            elif process.resources and process.state == 'waiting':
                process.state = 'running'
            
            # Update deadlock risk
            process.deadlock_risk = min(process.deadlock_risk + random.uniform(0.0, 0.01), 1.0)
    
    def run_sdd_analysis(self, num_iterations: int = 100) -> Dict:
        """Run the complete SDD analysis"""
        print("Starting Simplified SDD (Scalable Deadlock Detection) Analysis...")
        print(f"System: {self.num_processes} processes, {self.num_resources} resources")
        print("=" * 60)
        
        start_time = time.time()
        results = {
            'traditional_deadlocks': 0,
            'ml_deadlocks': 0,
            'ml_agreements': 0,
            'total_iterations': 0,
            'execution_time': 0.0,
            'performance_metrics': {},
            'deadlock_details': []
        }
        
        for iteration in range(num_iterations):
            # Simulate system evolution
            self._simulate_system_evolution(10)  # 10 steps per iteration
            
            # Traditional deadlock detection
            traditional_deadlock, traditional_info = self._detect_deadlock_traditional()
            
            # ML deadlock detection
            ml_deadlock, ml_probability = self._detect_deadlock_ml()
            
            # Update ML model
            self.gnn.update_weights(
                self.gnn.extract_features(self.gts.graph, self.processes, self.resources),
                traditional_deadlock,
                ml_deadlock
            )
            
            # Count detections
            if traditional_deadlock:
                results['traditional_deadlocks'] += 1
            
            if ml_deadlock:
                results['ml_deadlocks'] += 1
            
            # Check agreement
            if traditional_deadlock == ml_deadlock:
                results['ml_agreements'] += 1
            
            # Record deadlock details
            if traditional_deadlock or ml_deadlock:
                deadlock_detail = {
                    'iteration': iteration,
                    'traditional_detected': traditional_deadlock,
                    'ml_detected': ml_deadlock,
                    'ml_probability': ml_probability,
                    'traditional_info': traditional_info,
                    'graph_features': self.gts.get_graph_features()
                }
                results['deadlock_details'].append(deadlock_detail)
            
            # Show progress
            if iteration % 20 == 0:
                print(f"Iteration {iteration}/{num_iterations} - "
                      f"Traditional: {results['traditional_deadlocks']}, "
                      f"ML: {results['ml_deadlocks']}, "
                      f"Agreement: {results['ml_agreements']}")
        
        end_time = time.time()
        results['total_iterations'] = num_iterations
        results['execution_time'] = end_time - start_time
        
        # Calculate performance metrics
        results['performance_metrics'] = {
            'ml_accuracy': results['ml_agreements'] / num_iterations if num_iterations > 0 else 0,
            'traditional_detection_rate': results['traditional_deadlocks'] / num_iterations if num_iterations > 0 else 0,
            'ml_detection_rate': results['ml_deadlocks'] / num_iterations if num_iterations > 0 else 0,
            'avg_ml_probability': np.mean([d['ml_probability'] for d in results['deadlock_details']]) if results['deadlock_details'] else 0
        }
        
        print(f"\n✅ Simplified SDD Analysis completed!")
        print(f"⏱️  Execution time: {results['execution_time']:.2f} seconds")
        print(f"🔄 Total iterations: {results['total_iterations']}")
        print(f"🔍 Traditional deadlocks: {results['traditional_deadlocks']}")
        print(f"🧠 ML deadlocks: {results['ml_deadlocks']}")
        print(f"📊 ML accuracy: {results['performance_metrics']['ml_accuracy']:.3f}")
        
        return results
    
    def visualize_system_state(self, save_path: str = None):
        """Visualize the current system state"""
        plt.figure(figsize=(15, 10))
        
        # Create subplot for graph
        ax1 = plt.subplot(2, 2, 1)
        
        # Color nodes based on type
        node_colors = []
        node_sizes = []
        
        for node in self.gts.graph.nodes():
            if node.startswith("process_"):
                node_colors.append('lightblue')
                node_sizes.append(300)
            else:  # resource
                node_colors.append('lightgreen')
                node_sizes.append(200)
        
        # Draw graph
        pos = nx.spring_layout(self.gts.graph, k=3, iterations=50)
        nx.draw(self.gts.graph, pos, 
                node_color=node_colors, 
                node_size=node_sizes,
                with_labels=False, 
                arrows=True,
                edge_color='gray',
                alpha=0.7,
                ax=ax1)
        
        ax1.set_title(f"Simplified SDD System Graph - {self.num_processes} Processes, {self.num_resources} Resources")
        
        # Process state distribution
        ax2 = plt.subplot(2, 2, 2)
        states = ['Running', 'Waiting']
        counts = [
            sum(1 for p in self.processes if p.state == 'running'),
            sum(1 for p in self.processes if p.state == 'waiting')
        ]
        ax2.bar(states, counts, color=['lightblue', 'orange'])
        ax2.set_title('Process State Distribution')
        ax2.set_ylabel('Count')
        
        # Resource status
        ax3 = plt.subplot(2, 2, 3)
        resource_states = ['Available', 'Occupied']
        resource_counts = [
            sum(1 for r in self.resources if r.available),
            sum(1 for r in self.resources if not r.available)
        ]
        ax3.bar(resource_states, resource_counts, color=['lightgreen', 'red'])
        ax3.set_title('Resource Status')
        ax3.set_ylabel('Count')
        
        # System statistics
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        
        features = self.gts.get_graph_features()
        stats_text = f"""
Simplified SDD System Statistics:
• Number of processes: {self.num_processes}
• Number of resources: {self.num_resources}
• Graph nodes: {features['num_nodes']}
• Graph edges: {features['num_edges']}
• Graph density: {features['density']:.3f}
• Strongly connected: {features['is_strongly_connected']}
• Clustering coefficient: {features['clustering_coefficient']:.3f}
• Average deadlock risk: {np.mean([p.deadlock_risk for p in self.processes]):.3f}
        """
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Simplified SDD visualization saved to {save_path}")
        
        plt.show()
    
    def export_results(self, filename: str = "sdd_simplified_results.json"):
        """Export SDD analysis results"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.performance_metrics, f, indent=2, ensure_ascii=False)
        print(f"Simplified SDD results saved to {filename}")

def main():
    """Main function for simplified SDD approach demonstration"""
    print("Simplified SDD (Scalable Deadlock Detection) Approach")
    print("Using Graph Transformation Systems and Simplified ML")
    print("=" * 60)
    
    # Create SDD detector
    detector = SDDDetector(num_processes=50, num_resources=30)
    
    # Run SDD analysis
    results = detector.run_sdd_analysis(num_iterations=100)
    
    # Display results
    print("\n" + "=" * 60)
    print("Simplified SDD Analysis Results:")
    print(f"✅ Traditional deadlocks detected: {results['traditional_deadlocks']}")
    print(f"🧠 ML deadlocks detected: {results['ml_deadlocks']}")
    print(f"📊 ML accuracy: {results['performance_metrics']['ml_accuracy']:.3f}")
    print(f"⏱️  Execution time: {results['execution_time']:.2f} seconds")
    
    # Visualize system
    detector.visualize_system_state("sdd_simplified_visualization.png")
    
    # Export results
    detector.export_results("sdd_simplified_results.json")
    
    return detector, results

if __name__ == "__main__":
    detector, results = main()
