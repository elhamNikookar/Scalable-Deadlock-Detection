"""
Colored Resource-Oriented Petri Nets with Neural Network-based Deadlock Control
Implementation based on Kaid et al. (2021) methodology
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
import random
from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque
import threading
import queue


class TokenColor(Enum):
    """Token colors in colored Petri nets"""
    RED = "red"
    BLUE = "blue"
    GREEN = "green"
    YELLOW = "yellow"
    PURPLE = "purple"
    ORANGE = "orange"
    BLACK = "black"
    WHITE = "white"


class ResourceType(Enum):
    """Types of resources in manufacturing systems"""
    MACHINE = "machine"
    ROBOT = "robot"
    CONVEYOR = "conveyor"
    STORAGE = "storage"
    TOOL = "tool"
    WORKPIECE = "workpiece"


class FaultType(Enum):
    """Types of faults in manufacturing systems"""
    MACHINE_FAILURE = "machine_failure"
    COMMUNICATION_ERROR = "communication_error"
    RESOURCE_CONFLICT = "resource_conflict"
    TIMEOUT = "timeout"
    SENSOR_ERROR = "sensor_error"
    POWER_FAILURE = "power_failure"


@dataclass
class ColoredToken:
    """Colored token in Petri net"""
    id: str
    color: TokenColor
    value: Any = None
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Place:
    """Place in colored Petri net"""
    id: str
    name: str
    capacity: int = 1
    tokens: List[ColoredToken] = field(default_factory=list)
    resource_type: Optional[ResourceType] = None
    is_input: bool = False
    is_output: bool = False
    
    def add_token(self, token: ColoredToken) -> bool:
        """Add token to place if capacity allows"""
        if len(self.tokens) < self.capacity:
            self.tokens.append(token)
            return True
        return False
    
    def remove_token(self, token_id: str) -> Optional[ColoredToken]:
        """Remove token by ID"""
        for i, token in enumerate(self.tokens):
            if token.id == token_id:
                return self.tokens.pop(i)
        return None
    
    def get_tokens_by_color(self, color: TokenColor) -> List[ColoredToken]:
        """Get all tokens of specific color"""
        return [token for token in self.tokens if token.color == color]


@dataclass
class Transition:
    """Transition in colored Petri net"""
    id: str
    name: str
    input_places: List[str] = field(default_factory=list)
    output_places: List[str] = field(default_factory=list)
    guard_function: Optional[callable] = None
    firing_condition: Optional[callable] = None
    is_enabled: bool = False
    firing_time: float = 0.0
    priority: int = 0
    
    def can_fire(self, places: Dict[str, Place]) -> bool:
        """Check if transition can fire"""
        if self.guard_function:
            return self.guard_function(places)
        
        # Default firing condition: all input places have tokens
        for place_id in self.input_places:
            if place_id not in places or len(places[place_id].tokens) == 0:
                return False
        return True
    
    def fire(self, places: Dict[str, Place]) -> bool:
        """Fire the transition"""
        if not self.can_fire(places):
            return False
        
        # Remove tokens from input places
        for place_id in self.input_places:
            if place_id in places and places[place_id].tokens:
                places[place_id].tokens.pop(0)
        
        # Add tokens to output places
        for place_id in self.output_places:
            if place_id in places:
                # Create new token
                new_token = ColoredToken(
                    id=f"token_{int(time.time() * 1000)}",
                    color=TokenColor.BLUE,  # Default color
                    timestamp=time.time()
                )
                places[place_id].add_token(new_token)
        
        self.firing_time = time.time()
        return True


class ColoredResourceOrientedPetriNet:
    """
    Colored Resource-Oriented Petri Net implementation
    Based on Kaid et al. (2021) methodology
    """
    
    def __init__(self):
        self.places: Dict[str, Place] = {}
        self.transitions: Dict[str, Transition] = {}
        self.arcs: List[Tuple[str, str, str]] = []  # (source, target, type)
        self.marking_history: List[Dict[str, int]] = []
        self.firing_sequence: List[str] = []
        self.deadlock_states: List[Dict[str, int]] = []
        
    def add_place(self, place: Place):
        """Add place to Petri net"""
        self.places[place.id] = place
    
    def add_transition(self, transition: Transition):
        """Add transition to Petri net"""
        self.transitions[transition.id] = transition
    
    def add_arc(self, source: str, target: str, arc_type: str = "normal"):
        """Add arc between place and transition"""
        self.arcs.append((source, target, arc_type))
        
        # Update transition's input/output places
        if source in self.places and target in self.transitions:
            # Place to transition arc
            if source not in self.transitions[target].input_places:
                self.transitions[target].input_places.append(source)
        elif source in self.transitions and target in self.places:
            # Transition to place arc
            if target not in self.transitions[source].output_places:
                self.transitions[source].output_places.append(target)
    
    def get_current_marking(self) -> Dict[str, int]:
        """Get current marking of the Petri net"""
        marking = {}
        for place_id, place in self.places.items():
            marking[place_id] = len(place.tokens)
        return marking
    
    def detect_deadlock(self) -> Tuple[bool, List[str]]:
        """
        Detect deadlock using colored Petri net analysis
        """
        deadlocked_transitions = []
        
        # Check if any transition can fire
        for transition_id, transition in self.transitions.items():
            if not transition.can_fire(self.places):
                deadlocked_transitions.append(transition_id)
        
        # If all transitions are deadlocked, we have a deadlock
        is_deadlock = len(deadlocked_transitions) == len(self.transitions)
        
        if is_deadlock:
            self.deadlock_states.append(self.get_current_marking())
        
        return is_deadlock, deadlocked_transitions
    
    def simulate_step(self) -> bool:
        """Simulate one step of the Petri net"""
        # Find enabled transitions
        enabled_transitions = []
        for transition_id, transition in self.transitions.items():
            if transition.can_fire(self.places):
                enabled_transitions.append((transition_id, transition.priority))
        
        if not enabled_transitions:
            return False  # No enabled transitions
        
        # Sort by priority (higher priority first)
        enabled_transitions.sort(key=lambda x: x[1], reverse=True)
        
        # Fire the highest priority enabled transition
        transition_id, _ = enabled_transitions[0]
        transition = self.transitions[transition_id]
        
        if transition.fire(self.places):
            self.firing_sequence.append(transition_id)
            self.marking_history.append(self.get_current_marking())
            return True
        
        return False
    
    def run_simulation(self, max_steps: int = 1000) -> Dict[str, Any]:
        """Run simulation of the Petri net"""
        start_time = time.time()
        steps = 0
        
        while steps < max_steps:
            if not self.simulate_step():
                break
            steps += 1
        
        simulation_time = time.time() - start_time
        
        # Detect final deadlock state
        is_deadlock, deadlocked_transitions = self.detect_deadlock()
        
        return {
            'simulation_steps': steps,
            'simulation_time': simulation_time,
            'final_deadlock': is_deadlock,
            'deadlocked_transitions': deadlocked_transitions,
            'firing_sequence': self.firing_sequence,
            'marking_history': self.marking_history,
            'deadlock_states': self.deadlock_states
        }


class NeuralNetworkController:
    """
    Neural Network-based Deadlock Controller
    Based on Kaid et al. (2021) methodology
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize neural network weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        
        # Training data
        self.training_data: List[Tuple[np.ndarray, np.ndarray]] = []
        self.accuracy_history: List[float] = []
        
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward propagation"""
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        
        return a2, a1, z1
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make prediction"""
        output, _, _ = self.forward(X)
        return output
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, learning_rate: float = 0.1):
        """Train the neural network"""
        for epoch in range(epochs):
            # Forward propagation
            output, a1, z1 = self.forward(X)
            
            # Calculate loss
            loss = np.mean((output - y) ** 2)
            
            # Backward propagation
            dz2 = output - y
            dW2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)
            
            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * self.sigmoid_derivative(z1)
            dW1 = np.dot(X.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)
            
            # Update weights
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            
            # Calculate accuracy
            predictions = (output > 0.5).astype(int)
            accuracy = np.mean(predictions == y)
            self.accuracy_history.append(accuracy)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    def detect_deadlock(self, petri_net_state: np.ndarray) -> Tuple[bool, float]:
        """Detect deadlock using neural network"""
        prediction = self.predict(petri_net_state.reshape(1, -1))
        confidence = float(prediction[0, 0])
        is_deadlock = confidence > 0.5
        
        return is_deadlock, confidence
    
    def suggest_control_action(self, petri_net_state: np.ndarray) -> str:
        """Suggest control action based on neural network prediction"""
        prediction = self.predict(petri_net_state.reshape(1, -1))
        confidence = float(prediction[0, 0])
        
        if confidence > 0.8:
            return "immediate_intervention"
        elif confidence > 0.6:
            return "monitor_closely"
        elif confidence > 0.4:
            return "preventive_action"
        else:
            return "no_action_needed"


class FaultDetector:
    """
    Fault Detection and Treatment System
    Based on Kaid et al. (2021) methodology
    """
    
    def __init__(self):
        self.fault_history: List[Dict[str, Any]] = []
        self.fault_patterns: Dict[FaultType, List[Dict]] = defaultdict(list)
        self.treatment_strategies: Dict[FaultType, List[str]] = {
            FaultType.MACHINE_FAILURE: ["restart_machine", "replace_component", "manual_intervention"],
            FaultType.COMMUNICATION_ERROR: ["reset_communication", "check_network", "restart_system"],
            FaultType.RESOURCE_CONFLICT: ["release_resources", "reassign_tasks", "priority_adjustment"],
            FaultType.TIMEOUT: ["extend_timeout", "retry_operation", "skip_task"],
            FaultType.SENSOR_ERROR: ["calibrate_sensor", "replace_sensor", "manual_override"],
            FaultType.POWER_FAILURE: ["backup_power", "graceful_shutdown", "emergency_procedures"]
        }
        
    def detect_fault(self, petri_net: ColoredResourceOrientedPetriNet, 
                    system_state: Dict[str, Any]) -> Tuple[bool, List[FaultType], Dict[str, Any]]:
        """Detect faults in the system"""
        detected_faults = []
        fault_info = {}
        
        # Check for machine failures
        if self._check_machine_failure(system_state):
            detected_faults.append(FaultType.MACHINE_FAILURE)
            fault_info[FaultType.MACHINE_FAILURE] = {
                'severity': 'high',
                'affected_components': ['machine_1', 'machine_2'],
                'timestamp': time.time()
            }
        
        # Check for communication errors
        if self._check_communication_error(system_state):
            detected_faults.append(FaultType.COMMUNICATION_ERROR)
            fault_info[FaultType.COMMUNICATION_ERROR] = {
                'severity': 'medium',
                'affected_components': ['network'],
                'timestamp': time.time()
            }
        
        # Check for resource conflicts
        if self._check_resource_conflict(petri_net):
            detected_faults.append(FaultType.RESOURCE_CONFLICT)
            fault_info[FaultType.RESOURCE_CONFLICT] = {
                'severity': 'high',
                'affected_components': ['resource_pool'],
                'timestamp': time.time()
            }
        
        # Check for timeouts
        if self._check_timeout(system_state):
            detected_faults.append(FaultType.TIMEOUT)
            fault_info[FaultType.TIMEOUT] = {
                'severity': 'low',
                'affected_components': ['timer'],
                'timestamp': time.time()
            }
        
        # Check for sensor errors
        if self._check_sensor_error(system_state):
            detected_faults.append(FaultType.SENSOR_ERROR)
            fault_info[FaultType.SENSOR_ERROR] = {
                'severity': 'medium',
                'affected_components': ['sensor_array'],
                'timestamp': time.time()
            }
        
        # Check for power failures
        if self._check_power_failure(system_state):
            detected_faults.append(FaultType.POWER_FAILURE)
            fault_info[FaultType.POWER_FAILURE] = {
                'severity': 'critical',
                'affected_components': ['power_system'],
                'timestamp': time.time()
            }
        
        is_fault = len(detected_faults) > 0
        
        if is_fault:
            self.fault_history.append({
                'timestamp': time.time(),
                'faults': detected_faults,
                'fault_info': fault_info
            })
        
        return is_fault, detected_faults, fault_info
    
    def _check_machine_failure(self, system_state: Dict[str, Any]) -> bool:
        """Check for machine failure"""
        # Simulate machine failure detection
        return random.random() < 0.05  # 5% chance of machine failure
    
    def _check_communication_error(self, system_state: Dict[str, Any]) -> bool:
        """Check for communication error"""
        # Simulate communication error detection
        return random.random() < 0.03  # 3% chance of communication error
    
    def _check_resource_conflict(self, petri_net: ColoredResourceOrientedPetriNet) -> bool:
        """Check for resource conflict"""
        # Check if multiple transitions are competing for same resources
        resource_usage = defaultdict(int)
        for transition in petri_net.transitions.values():
            for place_id in transition.input_places:
                if place_id in petri_net.places:
                    place = petri_net.places[place_id]
                    if place.resource_type:
                        resource_usage[place.resource_type] += 1
        
        # If any resource is over-utilized, there's a conflict
        return any(count > 2 for count in resource_usage.values())
    
    def _check_timeout(self, system_state: Dict[str, Any]) -> bool:
        """Check for timeout"""
        # Simulate timeout detection
        return random.random() < 0.02  # 2% chance of timeout
    
    def _check_sensor_error(self, system_state: Dict[str, Any]) -> bool:
        """Check for sensor error"""
        # Simulate sensor error detection
        return random.random() < 0.04  # 4% chance of sensor error
    
    def _check_power_failure(self, system_state: Dict[str, Any]) -> bool:
        """Check for power failure"""
        # Simulate power failure detection
        return random.random() < 0.01  # 1% chance of power failure
    
    def treat_fault(self, fault_type: FaultType, fault_info: Dict[str, Any]) -> Tuple[bool, str]:
        """Treat detected fault"""
        treatment_strategies = self.treatment_strategies.get(fault_type, ["unknown_treatment"])
        
        # Select appropriate treatment strategy
        if fault_info.get('severity') == 'critical':
            treatment = treatment_strategies[-1]  # Use most aggressive treatment
        elif fault_info.get('severity') == 'high':
            treatment = treatment_strategies[1] if len(treatment_strategies) > 1 else treatment_strategies[0]
        else:
            treatment = treatment_strategies[0]  # Use least aggressive treatment
        
        # Simulate treatment success
        success_rate = 0.8 if fault_info.get('severity') != 'critical' else 0.6
        success = random.random() < success_rate
        
        return success, treatment


class ReconfigurableManufacturingSystem:
    """
    Reconfigurable Manufacturing System
    Based on Kaid et al. (2021) methodology
    """
    
    def __init__(self):
        self.petri_net = ColoredResourceOrientedPetriNet()
        self.neural_controller = NeuralNetworkController(input_size=10, hidden_size=20, output_size=1)
        self.fault_detector = FaultDetector()
        self.system_state: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, float] = {}
        
    def setup_manufacturing_system(self):
        """Setup a sample manufacturing system"""
        # Create places (resources)
        places = [
            Place("P1", "Raw Material Storage", capacity=5, resource_type=ResourceType.STORAGE, is_input=True),
            Place("P2", "Machine 1", capacity=1, resource_type=ResourceType.MACHINE),
            Place("P3", "Machine 2", capacity=1, resource_type=ResourceType.MACHINE),
            Place("P4", "Robot", capacity=1, resource_type=ResourceType.ROBOT),
            Place("P5", "Conveyor", capacity=3, resource_type=ResourceType.CONVEYOR),
            Place("P6", "Tool Station", capacity=2, resource_type=ResourceType.TOOL),
            Place("P7", "Quality Check", capacity=1, resource_type=ResourceType.MACHINE),
            Place("P8", "Finished Product Storage", capacity=10, resource_type=ResourceType.STORAGE, is_output=True)
        ]
        
        for place in places:
            self.petri_net.add_place(place)
        
        # Create transitions (operations)
        transitions = [
            Transition("T1", "Load Material", input_places=["P1"], output_places=["P2"], priority=1),
            Transition("T2", "Process Part 1", input_places=["P2"], output_places=["P3"], priority=2),
            Transition("T3", "Process Part 2", input_places=["P3"], output_places=["P4"], priority=2),
            Transition("T4", "Robot Operation", input_places=["P4"], output_places=["P5"], priority=3),
            Transition("T5", "Conveyor Transport", input_places=["P5"], output_places=["P6"], priority=1),
            Transition("T6", "Tool Operation", input_places=["P6"], output_places=["P7"], priority=2),
            Transition("T7", "Quality Check", input_places=["P7"], output_places=["P8"], priority=1)
        ]
        
        for transition in transitions:
            self.petri_net.add_transition(transition)
        
        # Add arcs
        arcs = [
            ("P1", "T1"), ("T1", "P2"),
            ("P2", "T2"), ("T2", "P3"),
            ("P3", "T3"), ("T3", "P4"),
            ("P4", "T4"), ("T4", "P5"),
            ("P5", "T5"), ("T5", "P6"),
            ("P6", "T6"), ("T6", "P7"),
            ("P7", "T7"), ("T7", "P8")
        ]
        
        for source, target in arcs:
            self.petri_net.add_arc(source, target)
        
        # Initialize with tokens
        for i in range(3):
            token = ColoredToken(f"token_{i}", TokenColor.RED, value=f"material_{i}")
            self.petri_net.places["P1"].add_token(token)
    
    def run_manufacturing_simulation(self, duration: float = 60.0) -> Dict[str, Any]:
        """Run manufacturing system simulation"""
        print("Starting manufacturing system simulation...")
        
        start_time = time.time()
        simulation_steps = 0
        deadlock_count = 0
        fault_count = 0
        successful_treatments = 0
        
        while time.time() - start_time < duration:
            # Update system state
            self.system_state = {
                'timestamp': time.time(),
                'marking': self.petri_net.get_current_marking(),
                'enabled_transitions': len([t for t in self.petri_net.transitions.values() if t.can_fire(self.petri_net.places)])
            }
            
            # Detect deadlock using neural network
            marking_array = np.array(list(self.petri_net.get_current_marking().values()))
            is_deadlock, confidence = self.neural_controller.detect_deadlock(marking_array)
            
            if is_deadlock:
                deadlock_count += 1
                print(f"Deadlock detected with confidence: {confidence:.3f}")
                
                # Suggest control action
                action = self.neural_controller.suggest_control_action(marking_array)
                print(f"Suggested action: {action}")
            
            # Detect faults
            is_fault, faults, fault_info = self.fault_detector.detect_fault(self.petri_net, self.system_state)
            
            if is_fault:
                fault_count += 1
                print(f"Faults detected: {[f.value for f in faults]}")
                
                # Treat faults
                for fault_type in faults:
                    success, treatment = self.fault_detector.treat_fault(fault_type, fault_info[fault_type])
                    if success:
                        successful_treatments += 1
                        print(f"Successfully treated {fault_type.value} using {treatment}")
                    else:
                        print(f"Failed to treat {fault_type.value}")
            
            # Run Petri net simulation step
            if self.petri_net.simulate_step():
                simulation_steps += 1
            
            time.sleep(0.1)  # Small delay for demonstration
        
        # Calculate performance metrics
        simulation_time = time.time() - start_time
        throughput = simulation_steps / simulation_time
        deadlock_rate = deadlock_count / simulation_steps if simulation_steps > 0 else 0
        fault_rate = fault_count / simulation_steps if simulation_steps > 0 else 0
        treatment_success_rate = successful_treatments / fault_count if fault_count > 0 else 0
        
        self.performance_metrics = {
            'simulation_time': simulation_time,
            'simulation_steps': simulation_steps,
            'throughput': throughput,
            'deadlock_count': deadlock_count,
            'deadlock_rate': deadlock_rate,
            'fault_count': fault_count,
            'fault_rate': fault_rate,
            'successful_treatments': successful_treatments,
            'treatment_success_rate': treatment_success_rate
        }
        
        print(f"Simulation completed!")
        print(f"Total steps: {simulation_steps}")
        print(f"Deadlocks detected: {deadlock_count}")
        print(f"Faults detected: {fault_count}")
        print(f"Successful treatments: {successful_treatments}")
        
        return self.performance_metrics


def demonstrate_kaid_methodology():
    """Demonstrate Kaid et al. (2021) methodology"""
    print("=" * 80)
    print("KAID ET AL. (2021) METHODOLOGY DEMONSTRATION")
    print("Colored Resource-Oriented Petri Nets with Neural Network Control")
    print("=" * 80)
    
    # Create manufacturing system
    manufacturing_system = ReconfigurableManufacturingSystem()
    manufacturing_system.setup_manufacturing_system()
    
    # Train neural network controller
    print("\nTraining neural network controller...")
    
    # Generate training data
    X_train = np.random.rand(100, 10)  # Random Petri net states
    y_train = np.random.randint(0, 2, (100, 1)).astype(float)  # Random deadlock labels
    
    manufacturing_system.neural_controller.train(X_train, y_train, epochs=500, learning_rate=0.1)
    
    # Run manufacturing simulation
    print("\nRunning manufacturing simulation...")
    performance_metrics = manufacturing_system.run_manufacturing_simulation(duration=30.0)
    
    # Print results
    print(f"\nPerformance Metrics:")
    for metric, value in performance_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return manufacturing_system, performance_metrics


if __name__ == "__main__":
    system, metrics = demonstrate_kaid_methodology()
