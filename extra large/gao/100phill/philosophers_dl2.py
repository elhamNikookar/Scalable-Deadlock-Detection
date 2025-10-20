"""
100 Philosophers Problem with Communication Deadlock Detection
Applying Gao et al. (2025) dl² methodology to dining philosophers
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
import random
import threading
import queue
from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

# Import dl² methodology
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dl2_deadlock_detector import (
    CommunicationType, OperationState, CommunicationOperation, 
    Process, CommunicationGraph, DL2DeadlockDetector
)


class PhilosopherState(Enum):
    """States of philosophers"""
    THINKING = "thinking"
    HUNGRY = "hungry"
    EATING = "eating"
    WAITING = "waiting"
    COMMUNICATING = "communicating"


@dataclass
class Philosopher:
    """Enhanced philosopher with communication capabilities"""
    id: int
    state: PhilosopherState = PhilosopherState.THINKING
    left_fork: Optional[int] = None
    right_fork: Optional[int] = None
    communication_queue: List[CommunicationOperation] = field(default_factory=list)
    current_communication: Optional[str] = None
    neighbors: List[int] = field(default_factory=list)
    communication_history: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        self.left_fork = self.id
        self.right_fork = (self.id + 1) % 100  # Circular arrangement
        # Define neighbors (philosophers this one can communicate with)
        self.neighbors = [(self.id - 1) % 100, (self.id + 1) % 100]


@dataclass
class Fork:
    """Enhanced fork with communication metadata"""
    id: int
    owner: Optional[int] = None
    is_available: bool = True
    usage_count: int = 0
    communication_requests: List[int] = field(default_factory=list)
    last_communication_time: float = 0.0


class CommunicationPhilosophersSystem:
    """
    Dining Philosophers System with Communication Patterns
    Applies dl² methodology for deadlock detection
    """
    
    def __init__(self, num_philosophers: int = 100):
        self.num_philosophers = num_philosophers
        self.philosophers: Dict[int, Philosopher] = {}
        self.forks: Dict[int, Fork] = {}
        self.dl2_detector = DL2DeadlockDetector()
        self.communication_graph = CommunicationGraph()
        
        # Statistics
        self.deadlock_events: List[Tuple[float, List[int], Dict]] = []
        self.communication_events: List[Dict] = []
        self.resolution_events: List[Tuple[float, str]] = []
        
        # Initialize system
        self._initialize_system()
        
    def _initialize_system(self):
        """Initialize philosophers, forks, and communication graph"""
        # Create philosophers
        for i in range(self.num_philosophers):
            philosopher = Philosopher(i)
            self.philosophers[i] = philosopher
            
            # Add to communication graph as process
            self.communication_graph.add_process(i, device_id=i % 4)  # 4 device groups
        
        # Create forks
        for i in range(self.num_philosophers):
            fork = Fork(i)
            self.forks[i] = fork
            
            # Add fork as resource in communication graph
            self.communication_graph.add_resource(f"Fork_{i}", capacity=1)
    
    def create_communication_operation(self, philosopher_id: int, operation_type: CommunicationType, 
                                     target_philosophers: List[int], metadata: Dict = None) -> CommunicationOperation:
        """Create a communication operation between philosophers"""
        operation_id = f"phil_{philosopher_id}_{operation_type.value}_{int(time.time() * 1000)}"
        
        operation = CommunicationOperation(
            id=operation_id,
            operation_type=operation_type,
            source_rank=philosopher_id,
            target_ranks=target_philosophers,
            tensor_shape=(1,),  # Simple scalar communication
            metadata=metadata or {}
        )
        
        # Add to communication graph
        self.communication_graph.add_operation(operation)
        
        # Add to philosopher's queue
        self.philosophers[philosopher_id].communication_queue.append(operation)
        
        return operation
    
    def simulate_philosopher_communication(self, philosopher_id: int):
        """Simulate communication behavior for a philosopher"""
        philosopher = self.philosophers[philosopher_id]
        
        # Random communication events
        if random.random() < 0.1:  # 10% chance to communicate
            # Choose communication type
            comm_types = [CommunicationType.BROADCAST, CommunicationType.SEND, CommunicationType.RECV]
            comm_type = random.choice(comm_types)
            
            # Choose target philosophers (neighbors)
            target_philosophers = random.sample(philosopher.neighbors, 
                                             min(2, len(philosopher.neighbors)))
            
            # Create communication operation
            operation = self.create_communication_operation(
                philosopher_id, comm_type, target_philosophers,
                {'reason': 'coordination', 'timestamp': time.time()}
            )
            
            # Record communication event
            self.communication_events.append({
                'timestamp': time.time(),
                'philosopher_id': philosopher_id,
                'operation_id': operation.id,
                'operation_type': comm_type.value,
                'target_philosophers': target_philosophers
            })
    
    def try_eat_with_communication(self, philosopher_id: int) -> bool:
        """
        Enhanced eating attempt with communication patterns
        """
        philosopher = self.philosophers[philosopher_id]
        left_fork_id = philosopher.left_fork
        right_fork_id = philosopher.right_fork
        
        # Check if both forks are available
        left_fork = self.forks[left_fork_id]
        right_fork = self.forks[right_fork_id]
        
        if left_fork.is_available and right_fork.is_available:
            # Acquire both forks
            philosopher.state = PhilosopherState.EATING
            left_fork.owner = philosopher_id
            left_fork.is_available = False
            right_fork.owner = philosopher_id
            right_fork.is_available = False
            
            # Create communication operations for fork acquisition
            self.create_communication_operation(
                philosopher_id, CommunicationType.BROADCAST, philosopher.neighbors,
                {'action': 'fork_acquisition', 'forks': [left_fork_id, right_fork_id]}
            )
            
            return True
        else:
            # Philosopher must wait - create waiting communication
            philosopher.state = PhilosopherState.WAITING
            
            # Create communication operations for waiting
            waiting_targets = []
            if not left_fork.is_available:
                waiting_targets.append(left_fork.owner)
            if not right_fork.is_available:
                waiting_targets.append(right_fork.owner)
            
            if waiting_targets:
                self.create_communication_operation(
                    philosopher_id, CommunicationType.SEND, waiting_targets,
                    {'action': 'waiting_for_forks', 'requested_forks': [left_fork_id, right_fork_id]}
                )
            
            return False
    
    def finish_eating_with_communication(self, philosopher_id: int):
        """Enhanced eating completion with communication"""
        philosopher = self.philosophers[philosopher_id]
        left_fork_id = philosopher.left_fork
        right_fork_id = philosopher.right_fork
        
        # Release both forks
        philosopher.state = PhilosopherState.THINKING
        self.forks[left_fork_id].owner = None
        self.forks[left_fork_id].is_available = True
        self.forks[left_fork_id].usage_count += 1
        
        self.forks[right_fork_id].owner = None
        self.forks[right_fork_id].is_available = True
        self.forks[right_fork_id].usage_count += 1
        
        # Create communication operation for fork release
        self.create_communication_operation(
            philosopher_id, CommunicationType.BROADCAST, philosopher.neighbors,
            {'action': 'fork_release', 'forks': [left_fork_id, right_fork_id]}
        )
    
    def detect_communication_deadlock(self) -> Tuple[bool, List[Set[str]], Dict[str, Any]]:
        """
        Detect communication deadlocks using dl² methodology
        """
        # Build job description from current system state
        job_description = self._build_job_description()
        
        # Analyze using dl² detector
        result = self.dl2_detector.analyze_deep_learning_job(job_description)
        
        return result['deadlock_detected'], result['deadlocks'], result['analysis_info']
    
    def _build_job_description(self) -> Dict[str, Any]:
        """Build job description from current system state"""
        # Extract processes
        processes = []
        for philosopher_id, philosopher in self.philosophers.items():
            processes.append({
                'rank': philosopher_id,
                'device_id': philosopher_id % 4,
                'state': philosopher.state.value
            })
        
        # Extract operations
        operations = []
        for operation in self.communication_graph.operations.values():
            operations.append({
                'id': operation.id,
                'type': operation.operation_type.value,
                'source_rank': operation.source_rank,
                'target_ranks': operation.target_ranks,
                'tensor_shape': list(operation.tensor_shape),
                'state': operation.state.value,
                'metadata': operation.metadata
            })
        
        # Extract dependencies
        dependencies = []
        for operation_id, operation in self.communication_graph.operations.items():
            for dep in operation.dependencies:
                dependencies.append({
                    'operation_id': operation_id,
                    'depends_on': dep
                })
        
        return {
            'processes': processes,
            'operations': operations,
            'dependencies': dependencies
        }
    
    def create_deadlock_scenario(self):
        """Create a controlled deadlock scenario for testing"""
        print("Creating controlled deadlock scenario with communication patterns...")
        
        # Make all philosophers hungry and try to eat simultaneously
        for i in range(self.num_philosophers):
            philosopher = self.philosophers[i]
            philosopher.state = PhilosopherState.HUNGRY
            
            # Try to acquire left fork
            left_fork_id = philosopher.left_fork
            right_fork_id = philosopher.right_fork
            
            # Acquire left fork
            if self.forks[left_fork_id].is_available:
                self.forks[left_fork_id].owner = i
                self.forks[left_fork_id].is_available = False
                
                # Create communication operation
                self.create_communication_operation(
                    i, CommunicationType.BROADCAST, philosopher.neighbors,
                    {'action': 'left_fork_acquired', 'fork_id': left_fork_id}
                )
            
            # Try to acquire right fork (this will cause deadlock)
            if self.forks[right_fork_id].is_available:
                self.forks[right_fork_id].owner = i
                self.forks[right_fork_id].is_available = False
                
                # Create communication operation
                self.create_communication_operation(
                    i, CommunicationType.BROADCAST, philosopher.neighbors,
                    {'action': 'right_fork_acquired', 'fork_id': right_fork_id}
                )
            else:
                # Right fork not available, philosopher waits
                philosopher.state = PhilosopherState.WAITING
                
                # Create waiting communication
                self.create_communication_operation(
                    i, CommunicationType.SEND, [self.forks[right_fork_id].owner],
                    {'action': 'waiting_for_right_fork', 'fork_id': right_fork_id}
                )
        
        print("Deadlock scenario created with communication patterns!")
    
    def resolve_communication_deadlock(self, deadlocks: List[Set[str]], strategy: str = "preempt") -> bool:
        """Resolve communication deadlock using dl² recommendations"""
        if not deadlocks:
            return False
        
        # Get the first deadlock
        deadlock_operations = list(deadlocks[0])
        
        if strategy == "preempt":
            # Preempt one philosopher's communication
            for operation_id in deadlock_operations:
                if operation_id in self.communication_graph.operations:
                    operation = self.communication_graph.operations[operation_id]
                    philosopher_id = operation.source_rank
                    
                    # Force release one fork
                    philosopher = self.philosophers[philosopher_id]
                    left_fork_id = philosopher.left_fork
                    
                    if not self.forks[left_fork_id].is_available:
                        self.forks[left_fork_id].owner = None
                        self.forks[left_fork_id].is_available = True
                        
                        philosopher.state = PhilosopherState.THINKING
                        
                        # Create resolution communication
                        self.create_communication_operation(
                            philosopher_id, CommunicationType.BROADCAST, philosopher.neighbors,
                            {'action': 'deadlock_resolution', 'strategy': 'preempt'}
                        )
                        
                        self.resolution_events.append((time.time(), f"Preempted philosopher {philosopher_id}"))
                        return True
        
        return False
    
    def simulate_step(self, time_step: float = 0.1):
        """Simulate one step with communication patterns"""
        # Randomly select philosophers for communication
        philosophers_to_communicate = random.sample(list(self.philosophers.keys()), 
                                                min(10, self.num_philosophers))
        
        for philosopher_id in philosophers_to_communicate:
            philosopher = self.philosophers[philosopher_id]
            
            # Simulate communication
            self.simulate_philosopher_communication(philosopher_id)
            
            if philosopher.state == PhilosopherState.THINKING:
                # Randomly decide to get hungry
                if random.random() < 0.1:  # 10% chance to get hungry
                    philosopher.state = PhilosopherState.HUNGRY
            
            elif philosopher.state == PhilosopherState.HUNGRY:
                # Try to eat with communication
                success = self.try_eat_with_communication(philosopher_id)
                if not success:
                    philosopher.waiting_time = getattr(philosopher, 'waiting_time', 0) + time_step
            
            elif philosopher.state == PhilosopherState.EATING:
                # Randomly finish eating
                if random.random() < 0.2:  # 20% chance to finish eating
                    self.finish_eating_with_communication(philosopher_id)
                    philosopher.eating_time = getattr(philosopher, 'eating_time', 0) + time_step
        
        # Check for communication deadlock
        is_deadlock, deadlocks, analysis_info = self.detect_communication_deadlock()
        
        if is_deadlock and len(deadlocks) > 0:
            self.deadlock_events.append((time.time(), [], analysis_info))
            
            # Resolve deadlock
            self.resolve_communication_deadlock(deadlocks, strategy="preempt")
    
    def run_simulation(self, duration: float = 30.0, time_step: float = 0.1):
        """Run simulation with communication patterns"""
        print(f"Starting simulation with communication patterns for {duration} seconds...")
        start_time = time.time()
        
        while time.time() - start_time < duration:
            self.simulate_step(time_step)
            time.sleep(time_step * 0.01)  # Scale down for demonstration
        
        print(f"Simulation completed!")
        self._print_statistics()
    
    def _print_statistics(self):
        """Print simulation statistics"""
        print(f"\nCommunication Philosophers Statistics:")
        print(f"  Total philosophers: {self.num_philosophers}")
        print(f"  Communication events: {len(self.communication_events)}")
        print(f"  Deadlock events: {len(self.deadlock_events)}")
        print(f"  Resolution events: {len(self.resolution_events)}")
        print(f"  Total operations: {len(self.communication_graph.operations)}")
        
        # Count philosophers by state
        state_counts = {'thinking': 0, 'hungry': 0, 'eating': 0, 'waiting': 0, 'communicating': 0}
        for philosopher in self.philosophers.values():
            state_counts[philosopher.state.value] += 1
        
        print(f"  Philosopher states:")
        for state, count in state_counts.items():
            print(f"    {state}: {count}")


class PhilosophersVisualizer:
    """Visualization tools for communication philosophers"""
    
    def __init__(self, system: CommunicationPhilosophersSystem):
        self.system = system
    
    def plot_communication_patterns(self, save_path: str = None):
        """Plot communication patterns over time"""
        if not self.system.communication_events:
            print("No communication events recorded")
            return
        
        # Extract data
        timestamps = [event['timestamp'] for event in self.system.communication_events]
        operation_types = [event['operation_type'] for event in self.system.communication_events]
        
        # Count operations by type
        type_counts = {}
        for op_type in operation_types:
            type_counts[op_type] = type_counts.get(op_type, 0) + 1
        
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Communication types distribution
        plt.subplot(2, 2, 1)
        plt.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
        plt.title('Communication Types Distribution')
        
        # Plot 2: Communication events over time
        plt.subplot(2, 2, 2)
        plt.scatter(timestamps, range(len(timestamps)), alpha=0.6)
        plt.title('Communication Events Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('Event Index')
        
        # Plot 3: Communication graph
        plt.subplot(2, 2, 3)
        self.system.communication_graph.visualize_communication_graph()
        
        # Plot 4: Deadlock events
        plt.subplot(2, 2, 4)
        if self.system.deadlock_events:
            deadlock_times = [event[0] for event in self.system.deadlock_events]
            plt.scatter(deadlock_times, [1] * len(deadlock_times), color='red', alpha=0.7)
            plt.title('Deadlock Events')
            plt.xlabel('Timestamp')
            plt.ylabel('Deadlock')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_philosophers_communication_network(self, save_path: str = None):
        """Plot philosophers communication network"""
        plt.figure(figsize=(15, 15))
        
        # Create circular layout for philosophers
        angles = np.linspace(0, 2*np.pi, self.system.num_philosophers, endpoint=False)
        radius = 5
        
        x_positions = radius * np.cos(angles)
        y_positions = radius * np.sin(angles)
        
        # Color code by state
        colors = []
        for philosopher in self.system.philosophers.values():
            if philosopher.state == PhilosopherState.EATING:
                colors.append('green')
            elif philosopher.state == PhilosopherState.HUNGRY:
                colors.append('orange')
            elif philosopher.state == PhilosopherState.WAITING:
                colors.append('red')
            elif philosopher.state == PhilosopherState.COMMUNICATING:
                colors.append('purple')
            else:  # THINKING
                colors.append('blue')
        
        # Plot philosophers
        plt.scatter(x_positions, y_positions, c=colors, s=100, alpha=0.7)
        
        # Plot communication connections
        for event in self.system.communication_events[-50:]:  # Last 50 events
            source_id = event['philosopher_id']
            target_ids = event['target_philosophers']
            
            for target_id in target_ids:
                if source_id < len(x_positions) and target_id < len(x_positions):
                    plt.plot([x_positions[source_id], x_positions[target_id]], 
                            [y_positions[source_id], y_positions[target_id]], 
                            'gray', alpha=0.3, linewidth=0.5)
        
        # Add labels
        for i, (x, y) in enumerate(zip(x_positions, y_positions)):
            plt.annotate(f'P{i}', (x, y), ha='center', va='center', fontsize=6)
        
        plt.title('Philosophers Communication Network')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [
            plt.scatter([], [], c='green', s=100, label='Eating'),
            plt.scatter([], [], c='orange', s=100, label='Hungry'),
            plt.scatter([], [], c='red', s=100, label='Waiting'),
            plt.scatter([], [], c='purple', s=100, label='Communicating'),
            plt.scatter([], [], c='blue', s=100, label='Thinking')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def run_philosophers_dl2_experiment():
    """Run comprehensive experiment applying dl² to philosophers"""
    print("=" * 80)
    print("100 PHILOSOPHERS WITH dl² COMMUNICATION DEADLOCK DETECTION")
    print("Applying Gao et al. (2025) methodology")
    print("=" * 80)
    
    # Create communication philosophers system
    print("Creating communication philosophers system...")
    system = CommunicationPhilosophersSystem(num_philosophers=100)
    
    # Test 1: Create deadlock scenario
    print("\n1. Creating deadlock scenario with communication patterns...")
    system.create_deadlock_scenario()
    
    # Test 2: Detect communication deadlock
    print("\n2. Detecting communication deadlocks using dl²...")
    is_deadlock, deadlocks, analysis_info = system.detect_communication_deadlock()
    
    print(f"   Communication deadlock detected: {is_deadlock}")
    print(f"   Number of deadlocks: {len(deadlocks)}")
    print(f"   Analysis methods used: {len(analysis_info.get('analysis_methods', []))}")
    
    # Test 3: Run simulation
    print("\n3. Running simulation with communication patterns...")
    system.run_simulation(duration=20.0, time_step=0.1)
    
    # Test 4: Analyze communication patterns
    print("\n4. Analyzing communication patterns...")
    job_description = system._build_job_description()
    dl2_result = system.dl2_detector.analyze_deep_learning_job(job_description)
    
    print(f"   dl² Analysis Results:")
    print(f"     Deadlock detected: {dl2_result['deadlock_detected']}")
    print(f"     Analysis time: {dl2_result['performance_metrics']['analysis_time']:.4f}s")
    print(f"     Operations analyzed: {dl2_result['performance_metrics']['operations_analyzed']}")
    
    # Print recommendations
    print(f"\n   Recommendations:")
    for i, rec in enumerate(dl2_result['recommendations'], 1):
        print(f"     {i}. {rec}")
    
    # Test 5: Create visualizations
    print("\n5. Creating visualizations...")
    visualizer = PhilosophersVisualizer(system)
    
    try:
        visualizer.plot_communication_patterns("philosophers_communication_patterns.png")
        visualizer.plot_philosophers_communication_network("philosophers_communication_network.png")
        print("   Visualizations saved!")
    except Exception as e:
        print(f"   Error creating visualizations: {e}")
    
    return system, dl2_result


def main():
    """Main function"""
    try:
        system, result = run_philosophers_dl2_experiment()
        
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        print(f"\nFinal Results:")
        print(f"  Total philosophers: {system.num_philosophers}")
        print(f"  Communication events: {len(system.communication_events)}")
        print(f"  Deadlock events: {len(system.deadlock_events)}")
        print(f"  dl² deadlock detection: {result['deadlock_detected']}")
        print(f"  Analysis time: {result['performance_metrics']['analysis_time']:.4f}s")
        
    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
