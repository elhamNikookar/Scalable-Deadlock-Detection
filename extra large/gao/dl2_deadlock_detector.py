"""
dl²: Detecting Communication Deadlocks in Deep Learning Jobs
Implementation based on Gao et al. (2025) methodology
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import threading
import queue
from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from collections import defaultdict, deque
import ast
import inspect


class CommunicationType(Enum):
    """Types of communication operations in deep learning"""
    ALLREDUCE = "allreduce"
    ALLGATHER = "allgather"
    BROADCAST = "broadcast"
    SCATTER = "scatter"
    GATHER = "gather"
    REDUCE_SCATTER = "reduce_scatter"
    SEND = "send"
    RECV = "recv"
    BARRIER = "barrier"


class OperationState(Enum):
    """States of communication operations"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass
class CommunicationOperation:
    """Represents a communication operation in deep learning"""
    id: str
    operation_type: CommunicationType
    source_rank: int
    target_ranks: List[int]
    tensor_shape: Tuple[int, ...]
    state: OperationState = OperationState.PENDING
    start_time: float = 0.0
    end_time: float = 0.0
    dependencies: Set[str] = field(default_factory=set)
    blocking_operations: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Process:
    """Represents a process/rank in distributed deep learning"""
    rank: int
    device_id: int
    operations: List[CommunicationOperation] = field(default_factory=list)
    current_operation: Optional[str] = None
    blocked_operations: Set[str] = field(default_factory=set)
    completed_operations: Set[str] = field(default_factory=set)


class CommunicationGraph:
    """
    Communication graph for deep learning jobs
    Based on dl² methodology
    """
    
    def __init__(self):
        self.processes: Dict[int, Process] = {}
        self.operations: Dict[str, CommunicationOperation] = {}
        self.graph = nx.DiGraph()
        self.communication_patterns: Dict[str, List[CommunicationOperation]] = defaultdict(list)
        self.deadlock_candidates: List[Set[str]] = []
        
    def add_process(self, rank: int, device_id: int = 0):
        """Add a process to the communication graph"""
        process = Process(rank=rank, device_id=device_id)
        self.processes[rank] = process
        self.graph.add_node(f"process_{rank}", node_type="process", rank=rank)
        
    def add_operation(self, operation: CommunicationOperation):
        """Add a communication operation"""
        self.operations[operation.id] = operation
        
        # Add operation node
        self.graph.add_node(operation.id, 
                          node_type="operation",
                          operation_type=operation.operation_type.value,
                          source_rank=operation.source_rank,
                          target_ranks=operation.target_ranks)
        
        # Add edges to target processes
        for target_rank in operation.target_ranks:
            if target_rank in self.processes:
                self.graph.add_edge(operation.id, f"process_{target_rank}",
                                  edge_type="communication")
        
        # Add to communication patterns
        pattern_key = f"{operation.operation_type.value}_{len(operation.target_ranks)}"
        self.communication_patterns[pattern_key].append(operation)
        
        # Add to process
        if operation.source_rank in self.processes:
            self.processes[operation.source_rank].operations.append(operation)
    
    def add_resource(self, resource_id: str, capacity: int = 1):
        """Add a resource to the communication graph"""
        self.graph.add_node(resource_id, node_type="resource", capacity=capacity)
    
    def add_dependency(self, operation_id: str, depends_on: str):
        """Add dependency between operations"""
        if operation_id in self.operations and depends_on in self.operations:
            self.operations[operation_id].dependencies.add(depends_on)
            self.graph.add_edge(depends_on, operation_id, edge_type="dependency")
    
    def detect_communication_deadlock(self) -> Tuple[bool, List[Set[str]], Dict[str, Any]]:
        """
        Detect communication deadlocks using dl² methodology
        """
        deadlock_info = {
            'detection_time': time.time(),
            'total_operations': len(self.operations),
            'total_processes': len(self.processes),
            'analysis_methods': []
        }
        
        # Method 1: Cycle detection in communication graph
        cycles = self._detect_communication_cycles()
        deadlock_info['cycle_detection'] = {
            'cycles_found': len(cycles),
            'cycles': cycles
        }
        
        # Method 2: Resource dependency analysis
        resource_deadlocks = self._analyze_resource_dependencies()
        deadlock_info['resource_analysis'] = resource_deadlocks
        
        # Method 3: Communication pattern analysis
        pattern_deadlocks = self._analyze_communication_patterns()
        deadlock_info['pattern_analysis'] = pattern_deadlocks
        
        # Method 4: Static analysis of operation sequences
        static_deadlocks = self._static_analysis()
        deadlock_info['static_analysis'] = static_deadlocks
        
        # Combine all deadlock candidates
        all_deadlocks = cycles + resource_deadlocks.get('deadlocks', []) + \
                       pattern_deadlocks.get('deadlocks', []) + static_deadlocks.get('deadlocks', [])
        
        # Remove duplicates and merge overlapping deadlocks
        unique_deadlocks = self._merge_deadlock_candidates(all_deadlocks)
        
        is_deadlock = len(unique_deadlocks) > 0
        deadlock_info['final_deadlocks'] = unique_deadlocks
        deadlock_info['deadlock_count'] = len(unique_deadlocks)
        
        return is_deadlock, unique_deadlocks, deadlock_info
    
    def _detect_communication_cycles(self) -> List[Set[str]]:
        """Detect cycles in the communication graph"""
        cycles = []
        
        try:
            # Find strongly connected components
            scc = list(nx.strongly_connected_components(self.graph))
            
            for component in scc:
                if len(component) > 1:
                    # Check if this component contains communication operations
                    operation_nodes = [node for node in component 
                                    if self.graph.nodes[node].get('node_type') == 'operation']
                    
                    if len(operation_nodes) > 1:
                        cycles.append(set(operation_nodes))
        
        except Exception as e:
            print(f"Error in cycle detection: {e}")
        
        return cycles
    
    def _analyze_resource_dependencies(self) -> Dict[str, Any]:
        """Analyze resource dependencies for deadlock detection"""
        analysis = {
            'resource_conflicts': [],
            'deadlocks': [],
            'blocking_chains': []
        }
        
        # Analyze each process for blocking operations
        for rank, process in self.processes.items():
            blocked_ops = []
            for operation in process.operations:
                if operation.state == OperationState.BLOCKED:
                    blocked_ops.append(operation.id)
            
            if blocked_ops:
                analysis['resource_conflicts'].append({
                    'process': rank,
                    'blocked_operations': blocked_ops
                })
        
        # Find blocking chains
        blocking_chains = self._find_blocking_chains()
        analysis['blocking_chains'] = blocking_chains
        
        # Convert blocking chains to deadlock candidates
        for chain in blocking_chains:
            if len(chain) > 2:  # Minimum chain length for deadlock
                analysis['deadlocks'].append(chain)
        
        return analysis
    
    def _analyze_communication_patterns(self) -> Dict[str, Any]:
        """Analyze communication patterns for potential deadlocks"""
        analysis = {
            'pattern_risks': {},
            'deadlocks': [],
            'risk_factors': []
        }
        
        # Analyze each communication pattern
        for pattern_name, operations in self.communication_patterns.items():
            risk_factors = []
            
            # Check for circular dependencies
            circular_deps = self._find_circular_dependencies(operations)
            if circular_deps:
                risk_factors.append(f"Circular dependencies: {len(circular_deps)}")
                analysis['deadlocks'].extend(circular_deps)
            
            # Check for resource contention
            contention = self._analyze_resource_contention(operations)
            if contention > 0.8:  # High contention threshold
                risk_factors.append(f"High resource contention: {contention:.2f}")
            
            # Check for asymmetric communication
            asymmetry = self._analyze_communication_asymmetry(operations)
            if asymmetry > 0.5:
                risk_factors.append(f"Communication asymmetry: {asymmetry:.2f}")
            
            analysis['pattern_risks'][pattern_name] = {
                'operation_count': len(operations),
                'risk_factors': risk_factors,
                'risk_score': len(risk_factors) / 3.0  # Normalize to [0,1]
            }
        
        return analysis
    
    def _static_analysis(self) -> Dict[str, Any]:
        """Static analysis of operation sequences"""
        analysis = {
            'deadlocks': [],
            'potential_issues': [],
            'optimization_suggestions': []
        }
        
        # Analyze operation sequences for each process
        for rank, process in self.processes.items():
            sequence = process.operations
            
            # Check for blocking patterns
            blocking_patterns = self._detect_blocking_patterns(sequence)
            if blocking_patterns:
                analysis['potential_issues'].extend(blocking_patterns)
            
            # Check for resource ordering issues
            ordering_issues = self._detect_ordering_issues(sequence)
            if ordering_issues:
                analysis['potential_issues'].extend(ordering_issues)
                analysis['optimization_suggestions'].append(
                    f"Process {rank}: Consider reordering operations to avoid deadlock"
                )
        
        # Cross-process analysis
        cross_process_issues = self._analyze_cross_process_patterns()
        analysis['potential_issues'].extend(cross_process_issues)
        
        return analysis
    
    def _find_blocking_chains(self) -> List[Set[str]]:
        """Find chains of blocking operations"""
        chains = []
        
        # Build dependency graph
        dep_graph = nx.DiGraph()
        for op_id, operation in self.operations.items():
            dep_graph.add_node(op_id)
            for dep in operation.dependencies:
                dep_graph.add_edge(dep, op_id)
        
        # Find paths that could lead to deadlock
        for op_id in self.operations:
            if self.operations[op_id].state == OperationState.BLOCKED:
                # Find all reachable operations
                reachable = set(nx.descendants(dep_graph, op_id))
                if reachable:
                    chains.append({op_id} | reachable)
        
        return chains
    
    def _find_circular_dependencies(self, operations: List[CommunicationOperation]) -> List[Set[str]]:
        """Find circular dependencies in a set of operations"""
        circular_deps = []
        
        # Build dependency graph for these operations
        dep_graph = nx.DiGraph()
        for op in operations:
            dep_graph.add_node(op.id)
            for dep in op.dependencies:
                if any(dep == other_op.id for other_op in operations):
                    dep_graph.add_edge(dep, op.id)
        
        # Find cycles
        try:
            cycles = list(nx.simple_cycles(dep_graph))
            for cycle in cycles:
                circular_deps.append(set(cycle))
        except:
            pass
        
        return circular_deps
    
    def _analyze_resource_contention(self, operations: List[CommunicationOperation]) -> float:
        """Analyze resource contention level"""
        if not operations:
            return 0.0
        
        # Count overlapping target ranks
        rank_usage = defaultdict(int)
        for op in operations:
            for rank in op.target_ranks:
                rank_usage[rank] += 1
        
        # Calculate contention as ratio of overused ranks
        max_usage = max(rank_usage.values()) if rank_usage else 0
        total_ranks = len(rank_usage)
        
        if total_ranks == 0:
            return 0.0
        
        return max_usage / total_ranks
    
    def _analyze_communication_asymmetry(self, operations: List[CommunicationOperation]) -> float:
        """Analyze communication asymmetry"""
        if not operations:
            return 0.0
        
        # Count operations by type
        type_counts = defaultdict(int)
        for op in operations:
            type_counts[op.operation_type] += 1
        
        # Calculate asymmetry as variance in type distribution
        if len(type_counts) <= 1:
            return 0.0
        
        counts = list(type_counts.values())
        mean_count = np.mean(counts)
        variance = np.var(counts)
        
        return variance / (mean_count + 1e-6)  # Normalize
    
    def _detect_blocking_patterns(self, sequence: List[CommunicationOperation]) -> List[str]:
        """Detect blocking patterns in operation sequence"""
        patterns = []
        
        for i, op in enumerate(sequence):
            if op.state == OperationState.BLOCKED:
                # Check if this creates a blocking chain
                blocking_chain = [op.id]
                
                # Look ahead for dependent operations
                for j in range(i + 1, len(sequence)):
                    next_op = sequence[j]
                    if op.id in next_op.dependencies:
                        blocking_chain.append(next_op.id)
                
                if len(blocking_chain) > 1:
                    patterns.append(f"Blocking chain detected: {' -> '.join(blocking_chain)}")
        
        return patterns
    
    def _detect_ordering_issues(self, sequence: List[CommunicationOperation]) -> List[str]:
        """Detect ordering issues that could lead to deadlock"""
        issues = []
        
        # Check for inconsistent resource ordering
        resource_order = {}
        
        for op in sequence:
            for rank in op.target_ranks:
                if rank in resource_order:
                    # Check if order is consistent
                    if op.id not in resource_order[rank]:
                        issues.append(f"Inconsistent resource ordering for rank {rank}")
                else:
                    resource_order[rank] = [op.id]
        
        return issues
    
    def _analyze_cross_process_patterns(self) -> List[str]:
        """Analyze patterns across processes"""
        issues = []
        
        # Check for cross-process dependencies
        cross_deps = []
        for op_id, operation in self.operations.items():
            for dep in operation.dependencies:
                if dep in self.operations:
                    dep_op = self.operations[dep]
                    if dep_op.source_rank != operation.source_rank:
                        cross_deps.append((dep_op.source_rank, operation.source_rank))
        
        if len(cross_deps) > len(self.processes):
            issues.append("High cross-process dependency complexity")
        
        return issues
    
    def _merge_deadlock_candidates(self, candidates: List[Set[str]]) -> List[Set[str]]:
        """Merge overlapping deadlock candidates"""
        if not candidates:
            return []
        
        merged = []
        candidates = [set(c) for c in candidates]
        
        for candidate in candidates:
            # Check if this candidate overlaps with any existing merged candidate
            merged_with_existing = False
            
            for i, existing in enumerate(merged):
                if candidate & existing:  # Overlap exists
                    merged[i] = candidate | existing
                    merged_with_existing = True
                    break
            
            if not merged_with_existing:
                merged.append(candidate)
        
        return merged
    
    def visualize_communication_graph(self, save_path: str = None):
        """Visualize the communication graph"""
        plt.figure(figsize=(15, 10))
        
        # Create layout
        pos = nx.spring_layout(self.graph, k=3, iterations=50)
        
        # Draw nodes
        process_nodes = [node for node in self.graph.nodes() 
                        if self.graph.nodes[node].get('node_type') == 'process']
        operation_nodes = [node for node in self.graph.nodes() 
                          if self.graph.nodes[node].get('node_type') == 'operation']
        
        # Draw process nodes
        nx.draw_networkx_nodes(self.graph, pos, nodelist=process_nodes,
                              node_color='lightblue', node_size=1000, alpha=0.8)
        
        # Draw operation nodes
        nx.draw_networkx_nodes(self.graph, pos, nodelist=operation_nodes,
                              node_color='lightcoral', node_size=500, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, alpha=0.6, edge_color='gray')
        
        # Add labels
        labels = {}
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('node_type') == 'process':
                labels[node] = f"P{self.graph.nodes[node].get('rank', '?')}"
            else:
                labels[node] = node[:8] + "..." if len(node) > 8 else node
        
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        plt.title("Communication Graph for Deep Learning Job")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class DL2DeadlockDetector:
    """
    Main dl² deadlock detector class
    Implements the methodology from Gao et al. (2025)
    """
    
    def __init__(self):
        self.communication_graph = CommunicationGraph()
        self.detection_history: List[Dict] = []
        self.performance_metrics: Dict[str, float] = {}
        
    def analyze_deep_learning_job(self, job_description: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a deep learning job for communication deadlocks
        """
        start_time = time.time()
        
        # Parse job description and build communication graph
        self._parse_job_description(job_description)
        
        # Detect deadlocks
        is_deadlock, deadlocks, analysis_info = self.communication_graph.detect_communication_deadlock()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(analysis_info)
        
        # Calculate performance metrics
        analysis_time = time.time() - start_time
        self.performance_metrics['analysis_time'] = analysis_time
        self.performance_metrics['operations_analyzed'] = len(self.communication_graph.operations)
        
        result = {
            'deadlock_detected': is_deadlock,
            'deadlocks': deadlocks,
            'analysis_info': analysis_info,
            'recommendations': recommendations,
            'performance_metrics': self.performance_metrics,
            'timestamp': time.time()
        }
        
        self.detection_history.append(result)
        return result
    
    def _parse_job_description(self, job_description: Dict[str, Any]):
        """Parse job description and build communication graph"""
        # Extract processes
        processes = job_description.get('processes', [])
        for proc_info in processes:
            rank = proc_info.get('rank', 0)
            device_id = proc_info.get('device_id', 0)
            self.communication_graph.add_process(rank, device_id)
        
        # Extract operations
        operations = job_description.get('operations', [])
        for op_info in operations:
            operation = CommunicationOperation(
                id=op_info.get('id', f"op_{len(self.communication_graph.operations)}"),
                operation_type=CommunicationType(op_info.get('type', 'allreduce')),
                source_rank=op_info.get('source_rank', 0),
                target_ranks=op_info.get('target_ranks', []),
                tensor_shape=tuple(op_info.get('tensor_shape', [])),
                metadata=op_info.get('metadata', {})
            )
            self.communication_graph.add_operation(operation)
        
        # Extract dependencies
        dependencies = job_description.get('dependencies', [])
        for dep_info in dependencies:
            operation_id = dep_info.get('operation_id')
            depends_on = dep_info.get('depends_on')
            if operation_id and depends_on:
                self.communication_graph.add_dependency(operation_id, depends_on)
    
    def _generate_recommendations(self, analysis_info: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Cycle detection recommendations
        cycles = analysis_info.get('cycle_detection', {}).get('cycles', [])
        if cycles:
            recommendations.append(f"Found {len(cycles)} communication cycles. Consider breaking cycles by reordering operations.")
        
        # Resource analysis recommendations
        resource_conflicts = analysis_info.get('resource_analysis', {}).get('resource_conflicts', [])
        if resource_conflicts:
            recommendations.append(f"Detected {len(resource_conflicts)} resource conflicts. Consider implementing timeout mechanisms.")
        
        # Pattern analysis recommendations
        pattern_risks = analysis_info.get('pattern_analysis', {}).get('pattern_risks', {})
        high_risk_patterns = [name for name, info in pattern_risks.items() 
                            if info.get('risk_score', 0) > 0.7]
        if high_risk_patterns:
            recommendations.append(f"High-risk communication patterns detected: {', '.join(high_risk_patterns)}")
        
        # Static analysis recommendations
        optimization_suggestions = analysis_info.get('static_analysis', {}).get('optimization_suggestions', [])
        recommendations.extend(optimization_suggestions)
        
        return recommendations
    
    def create_sample_deep_learning_job(self) -> Dict[str, Any]:
        """Create a sample deep learning job for testing"""
        job = {
            'processes': [
                {'rank': i, 'device_id': i % 4} for i in range(8)  # 8 processes, 4 GPUs
            ],
            'operations': [
                # AllReduce operations
                {'id': 'allreduce_1', 'type': 'allreduce', 'source_rank': 0, 'target_ranks': [0, 1, 2, 3], 'tensor_shape': [1024, 1024]},
                {'id': 'allreduce_2', 'type': 'allreduce', 'source_rank': 4, 'target_ranks': [4, 5, 6, 7], 'tensor_shape': [1024, 1024]},
                
                # Broadcast operations
                {'id': 'broadcast_1', 'type': 'broadcast', 'source_rank': 0, 'target_ranks': [1, 2, 3], 'tensor_shape': [512, 512]},
                {'id': 'broadcast_2', 'type': 'broadcast', 'source_rank': 4, 'target_ranks': [5, 6, 7], 'tensor_shape': [512, 512]},
                
                # Gather operations
                {'id': 'gather_1', 'type': 'gather', 'source_rank': 0, 'target_ranks': [1, 2, 3], 'tensor_shape': [256, 256]},
                {'id': 'gather_2', 'type': 'gather', 'source_rank': 4, 'target_ranks': [5, 6, 7], 'tensor_shape': [256, 256]},
            ],
            'dependencies': [
                {'operation_id': 'broadcast_1', 'depends_on': 'allreduce_1'},
                {'operation_id': 'gather_1', 'depends_on': 'broadcast_1'},
                {'operation_id': 'broadcast_2', 'depends_on': 'allreduce_2'},
                {'operation_id': 'gather_2', 'depends_on': 'broadcast_2'},
            ]
        }
        return job


def demonstrate_dl2_methodology():
    """Demonstrate the dl² methodology"""
    print("=" * 80)
    print("dl²: DETECTING COMMUNICATION DEADLOCKS IN DEEP LEARNING JOBS")
    print("Implementation based on Gao et al. (2025)")
    print("=" * 80)
    
    # Create dl² detector
    detector = DL2DeadlockDetector()
    
    # Create sample job
    print("Creating sample deep learning job...")
    sample_job = detector.create_sample_deep_learning_job()
    
    print(f"Job contains:")
    print(f"  Processes: {len(sample_job['processes'])}")
    print(f"  Operations: {len(sample_job['operations'])}")
    print(f"  Dependencies: {len(sample_job['dependencies'])}")
    
    # Analyze job
    print("\nAnalyzing job for communication deadlocks...")
    result = detector.analyze_deep_learning_job(sample_job)
    
    # Print results
    print(f"\nAnalysis Results:")
    print(f"  Deadlock detected: {result['deadlock_detected']}")
    print(f"  Number of deadlocks: {result['analysis_info']['deadlock_count']}")
    print(f"  Analysis time: {result['performance_metrics']['analysis_time']:.4f}s")
    
    # Print detailed analysis
    analysis_info = result['analysis_info']
    
    print(f"\nDetailed Analysis:")
    print(f"  Cycle detection: {analysis_info['cycle_detection']['cycles_found']} cycles found")
    print(f"  Resource conflicts: {len(analysis_info['resource_analysis']['resource_conflicts'])}")
    print(f"  Communication patterns: {len(analysis_info['pattern_analysis']['pattern_risks'])}")
    
    # Print recommendations
    print(f"\nRecommendations:")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Visualize communication graph
    print(f"\nCreating communication graph visualization...")
    detector.communication_graph.visualize_communication_graph("dl2_communication_graph.png")
    
    return detector, result


if __name__ == "__main__":
    detector, result = demonstrate_dl2_methodology()
