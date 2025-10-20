"""
Deadlock Detection Algorithm Implementation
Based on Salimi et al. (2020) methodology
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from collections import deque
import itertools


@dataclass
class Resource:
    """Represents a resource in the system"""
    id: str
    capacity: int = 1
    allocated: int = 0
    waiting_processes: List[str] = None
    
    def __post_init__(self):
        if self.waiting_processes is None:
            self.waiting_processes = []


@dataclass
class Process:
    """Represents a process in the system"""
    id: str
    allocated_resources: Set[str] = None
    requested_resources: Set[str] = None
    state: str = "running"  # running, waiting, blocked
    
    def __post_init__(self):
        if self.allocated_resources is None:
            self.allocated_resources = set()
        if self.requested_resources is None:
            self.requested_resources = set()


class DeadlockDetector:
    """
    Deadlock Detection Algorithm based on Salimi et al. (2020)
    Implements multiple deadlock detection strategies
    """
    
    def __init__(self):
        self.processes: Dict[str, Process] = {}
        self.resources: Dict[str, Resource] = {}
        self.wait_for_graph: nx.DiGraph = nx.DiGraph()
        self.allocation_matrix: Optional[np.ndarray] = None
        self.request_matrix: Optional[np.ndarray] = None
        self.available_resources: Optional[np.ndarray] = None
        
    def add_process(self, process: Process):
        """Add a process to the system"""
        self.processes[process.id] = process
        self.wait_for_graph.add_node(process.id, node_type='process')
        
    def add_resource(self, resource: Resource):
        """Add a resource to the system"""
        self.resources[resource.id] = resource
        self.wait_for_graph.add_node(resource.id, node_type='resource')
        
    def allocate_resource(self, process_id: str, resource_id: str) -> bool:
        """Allocate a resource to a process"""
        if process_id not in self.processes or resource_id not in self.resources:
            return False
            
        process = self.processes[process_id]
        resource = self.resources[resource_id]
        
        if resource.allocated < resource.capacity:
            resource.allocated += 1
            process.allocated_resources.add(resource_id)
            return True
        else:
            # Resource is not available, process must wait
            process.state = "waiting"
            resource.waiting_processes.append(process_id)
            process.requested_resources.add(resource_id)
            
            # Add edge to wait-for graph
            self.wait_for_graph.add_edge(process_id, resource_id, edge_type='waiting')
            return False
    
    def release_resource(self, process_id: str, resource_id: str) -> bool:
        """Release a resource from a process"""
        if process_id not in self.processes or resource_id not in self.resources:
            return False
            
        process = self.processes[process_id]
        resource = self.resources[resource_id]
        
        if resource_id in process.allocated_resources:
            resource.allocated -= 1
            process.allocated_resources.remove(resource_id)
            
            # Remove waiting edge
            if self.wait_for_graph.has_edge(process_id, resource_id):
                self.wait_for_graph.remove_edge(process_id, resource_id)
            
            # Grant resource to waiting process if any
            if resource.waiting_processes:
                waiting_process_id = resource.waiting_processes.pop(0)
                waiting_process = self.processes[waiting_process_id]
                waiting_process.state = "running"
                waiting_process.requested_resources.discard(resource_id)
                waiting_process.allocated_resources.add(resource_id)
                resource.allocated += 1
                
                # Remove waiting edge
                if self.wait_for_graph.has_edge(waiting_process_id, resource_id):
                    self.wait_for_graph.remove_edge(waiting_process_id, resource_id)
            
            return True
        return False
    
    def detect_deadlock_cycle_detection(self) -> Tuple[bool, List[str]]:
        """
        Detect deadlock using cycle detection in wait-for graph
        """
        try:
            # Find strongly connected components
            scc = list(nx.strongly_connected_components(self.wait_for_graph))
            
            for component in scc:
                if len(component) > 1:
                    # Check if this component forms a deadlock cycle
                    subgraph = self.wait_for_graph.subgraph(component)
                    if nx.is_strongly_connected(subgraph):
                        return True, list(component)
            
            return False, []
        except Exception as e:
            print(f"Error in cycle detection: {e}")
            return False, []
    
    def detect_deadlock_resource_allocation(self) -> Tuple[bool, List[str]]:
        """
        Detect deadlock using resource allocation graph analysis
        """
        deadlocked_processes = []
        
        # Check each process
        for process_id, process in self.processes.items():
            if process.state == "waiting":
                # Check if the process can be satisfied
                can_be_satisfied = True
                
                for resource_id in process.requested_resources:
                    resource = self.resources[resource_id]
                    if resource.allocated >= resource.capacity:
                        can_be_satisfied = False
                        break
                
                if not can_be_satisfied:
                    deadlocked_processes.append(process_id)
        
        return len(deadlocked_processes) > 0, deadlocked_processes
    
    def detect_deadlock_banker_algorithm(self) -> Tuple[bool, List[str]]:
        """
        Detect deadlock using Banker's algorithm approach
        """
        if not self.allocation_matrix is not None:
            self._build_matrices()
        
        if self.allocation_matrix is None:
            return False, []
        
        n_processes = len(self.processes)
        n_resources = len(self.resources)
        
        # Initialize work vector
        work = self.available_resources.copy()
        
        # Initialize finish vector
        finish = np.zeros(n_processes, dtype=bool)
        
        # Find a process that can be finished
        while True:
            found = False
            for i in range(n_processes):
                if not finish[i]:
                    # Check if process i can be satisfied
                    need = self.request_matrix[i] - self.allocation_matrix[i]
                    if np.all(need <= work):
                        # Process i can be finished
                        work += self.allocation_matrix[i]
                        finish[i] = True
                        found = True
                        break
            
            if not found:
                break
        
        # Check for deadlock
        deadlocked_processes = []
        for i in range(n_processes):
            if not finish[i]:
                process_id = list(self.processes.keys())[i]
                deadlocked_processes.append(process_id)
        
        return len(deadlocked_processes) > 0, deadlocked_processes
    
    def detect_deadlock_fuzzy_approach(self, fuzzy_threshold: float = 0.7) -> Tuple[bool, List[str], float]:
        """
        Detect deadlock using fuzzy logic approach
        Returns (is_deadlock, deadlocked_processes, confidence_score)
        """
        deadlock_probabilities = {}
        
        for process_id, process in self.processes.items():
            if process.state == "waiting":
                # Calculate deadlock probability for this process
                probability = self._calculate_deadlock_probability(process)
                deadlock_probabilities[process_id] = probability
        
        # Find processes with high deadlock probability
        deadlocked_processes = [
            pid for pid, prob in deadlock_probabilities.items() 
            if prob >= fuzzy_threshold
        ]
        
        # Calculate overall confidence
        if deadlock_probabilities:
            avg_probability = np.mean(list(deadlock_probabilities.values()))
            max_probability = np.max(list(deadlock_probabilities.values()))
            confidence_score = (avg_probability + max_probability) / 2
        else:
            confidence_score = 0.0
        
        is_deadlock = len(deadlocked_processes) > 0 and confidence_score >= fuzzy_threshold
        
        return is_deadlock, deadlocked_processes, confidence_score
    
    def _calculate_deadlock_probability(self, process: Process) -> float:
        """Calculate deadlock probability for a process using fuzzy logic"""
        probability = 0.0
        
        # Factor 1: Number of requested resources
        num_requested = len(process.requested_resources)
        factor1 = min(1.0, num_requested / 5.0)  # Normalize to [0,1]
        
        # Factor 2: Resource scarcity
        scarcity_factor = 0.0
        for resource_id in process.requested_resources:
            resource = self.resources[resource_id]
            scarcity = resource.allocated / resource.capacity
            scarcity_factor += scarcity
        factor2 = scarcity_factor / len(process.requested_resources) if process.requested_resources else 0.0
        
        # Factor 3: Waiting time (simplified)
        factor3 = 0.5  # Placeholder for waiting time
        
        # Combine factors using fuzzy logic
        probability = (factor1 * 0.4 + factor2 * 0.4 + factor3 * 0.2)
        
        return min(1.0, probability)
    
    def _build_matrices(self):
        """Build allocation, request, and available matrices for Banker's algorithm"""
        if not self.processes or not self.resources:
            return
        
        process_ids = list(self.processes.keys())
        resource_ids = list(self.resources.keys())
        
        n_processes = len(process_ids)
        n_resources = len(resources)
        
        # Initialize matrices
        self.allocation_matrix = np.zeros((n_processes, n_resources))
        self.request_matrix = np.zeros((n_processes, n_resources))
        self.available_resources = np.zeros(n_resources)
        
        # Fill allocation matrix
        for i, process_id in enumerate(process_ids):
            process = self.processes[process_id]
            for j, resource_id in enumerate(resource_ids):
                if resource_id in process.allocated_resources:
                    self.allocation_matrix[i, j] = 1
        
        # Fill request matrix
        for i, process_id in enumerate(process_ids):
            process = self.processes[process_id]
            for j, resource_id in enumerate(resource_ids):
                if resource_id in process.requested_resources:
                    self.request_matrix[i, j] = 1
        
        # Fill available resources
        for j, resource_id in enumerate(resource_ids):
            resource = self.resources[resource_id]
            self.available_resources[j] = resource.capacity - resource.allocated
    
    def resolve_deadlock(self, deadlocked_processes: List[str], strategy: str = "terminate") -> bool:
        """
        Resolve deadlock using various strategies
        """
        if strategy == "terminate":
            return self._terminate_processes(deadlocked_processes)
        elif strategy == "preempt":
            return self._preempt_resources(deadlocked_processes)
        elif strategy == "rollback":
            return self._rollback_processes(deadlocked_processes)
        else:
            return False
    
    def _terminate_processes(self, process_ids: List[str]) -> bool:
        """Terminate deadlocked processes"""
        for process_id in process_ids:
            if process_id in self.processes:
                process = self.processes[process_id]
                
                # Release all allocated resources
                for resource_id in list(process.allocated_resources):
                    self.release_resource(process_id, resource_id)
                
                # Remove from system
                del self.processes[process_id]
                if self.wait_for_graph.has_node(process_id):
                    self.wait_for_graph.remove_node(process_id)
        
        return True
    
    def _preempt_resources(self, process_ids: List[str]) -> bool:
        """Preempt resources from deadlocked processes"""
        for process_id in process_ids:
            if process_id in self.processes:
                process = self.processes[process_id]
                
                # Release one resource to break the cycle
                if process.allocated_resources:
                    resource_id = next(iter(process.allocated_resources))
                    self.release_resource(process_id, resource_id)
                    break
        
        return True
    
    def _rollback_processes(self, process_ids: List[str]) -> bool:
        """Rollback deadlocked processes to a previous safe state"""
        # Simplified rollback - just release all resources
        for process_id in process_ids:
            if process_id in self.processes:
                process = self.processes[process_id]
                
                # Release all allocated resources
                for resource_id in list(process.allocated_resources):
                    self.release_resource(process_id, resource_id)
                
                # Reset process state
                process.state = "running"
                process.requested_resources.clear()
        
        return True
    
    def get_system_state(self) -> Dict:
        """Get current system state"""
        return {
            'num_processes': len(self.processes),
            'num_resources': len(self.resources),
            'waiting_processes': len([p for p in self.processes.values() if p.state == "waiting"]),
            'graph_edges': self.wait_for_graph.number_of_edges(),
            'graph_nodes': self.wait_for_graph.number_of_nodes()
        }
    
    def visualize_wait_for_graph(self) -> nx.DiGraph:
        """Return the wait-for graph for visualization"""
        return self.wait_for_graph.copy()


class AdvancedDeadlockDetector(DeadlockDetector):
    """
    Advanced deadlock detector with additional features
    """
    
    def __init__(self):
        super().__init__()
        self.deadlock_history: List[Tuple[float, List[str]]] = []
        self.detection_threshold = 0.5
        
    def detect_deadlock_with_history(self) -> Tuple[bool, List[str], float]:
        """
        Detect deadlock considering historical patterns
        """
        # Get current deadlock detection
        is_deadlock, processes, confidence = self.detect_deadlock_fuzzy_approach()
        
        # Consider historical patterns
        if self.deadlock_history:
            recent_deadlocks = [entry for entry in self.deadlock_history[-5:] if entry[0] > 0.5]
            if len(recent_deadlocks) >= 3:
                # System has been experiencing frequent deadlocks
                confidence *= 1.2  # Increase confidence
        
        # Record current state
        self.deadlock_history.append((confidence, processes))
        
        return is_deadlock, processes, min(1.0, confidence)
    
    def predict_deadlock_probability(self) -> float:
        """
        Predict probability of future deadlock based on current state
        """
        if not self.deadlock_history:
            return 0.0
        
        # Analyze recent trends
        recent_confidences = [entry[0] for entry in self.deadlock_history[-10:]]
        
        if len(recent_confidences) < 3:
            return recent_confidences[-1] if recent_confidences else 0.0
        
        # Calculate trend
        trend = np.polyfit(range(len(recent_confidences)), recent_confidences, 1)[0]
        
        # Predict future probability
        current_prob = recent_confidences[-1]
        predicted_prob = current_prob + trend * 2  # Predict 2 steps ahead
        
        return max(0.0, min(1.0, predicted_prob))
