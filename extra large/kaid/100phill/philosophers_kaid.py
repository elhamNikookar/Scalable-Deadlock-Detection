"""
100 Philosophers Problem with Colored Resource-Oriented Petri Nets and Neural Network Control
Applying Kaid et al. (2021) methodology to dining philosophers
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import random
from typing import List, Dict, Tuple, Set, Any, Optional
import sys
import os

# Import Kaid methodology
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from colored_petri_nets import (
    ColoredResourceOrientedPetriNet, Place, Transition, ColoredToken,
    TokenColor, ResourceType, FaultType, NeuralNetworkController, FaultDetector
)


class PhilosophersPetriNet:
    """
    Dining Philosophers Problem modeled as Colored Resource-Oriented Petri Net
    Based on Kaid et al. (2021) methodology
    """
    
    def __init__(self, num_philosophers: int = 100):
        self.num_philosophers = num_philosophers
        self.petri_net = ColoredResourceOrientedPetriNet()
        self.neural_controller = NeuralNetworkController(
            input_size=num_philosophers * 3,  # Each philosopher has 3 states
            hidden_size=50,
            output_size=1
        )
        self.fault_detector = FaultDetector()
        
        # Statistics
        self.deadlock_events: List[Dict[str, Any]] = []
        self.fault_events: List[Dict[str, Any]] = []
        self.control_actions: List[Dict[str, Any]] = []
        
        # Initialize the Petri net
        self._setup_philosophers_petri_net()
        
    def _setup_philosophers_petri_net(self):
        """Setup Petri net for dining philosophers problem"""
        print(f"Setting up Petri net for {self.num_philosophers} philosophers...")
        
        # Create places for philosophers (thinking, hungry, eating states)
        for i in range(self.num_philosophers):
            # Thinking state
            thinking_place = Place(
                f"thinking_{i}", 
                f"Philosopher {i} Thinking",
                capacity=1,
                resource_type=ResourceType.MACHINE
            )
            thinking_place.add_token(ColoredToken(f"think_token_{i}", TokenColor.BLUE))
            self.petri_net.add_place(thinking_place)
            
            # Hungry state
            hungry_place = Place(
                f"hungry_{i}",
                f"Philosopher {i} Hungry", 
                capacity=1,
                resource_type=ResourceType.MACHINE
            )
            self.petri_net.add_place(hungry_place)
            
            # Eating state
            eating_place = Place(
                f"eating_{i}",
                f"Philosopher {i} Eating",
                capacity=1,
                resource_type=ResourceType.MACHINE
            )
            self.petri_net.add_place(eating_place)
        
        # Create places for forks
        for i in range(self.num_philosophers):
            fork_place = Place(
                f"fork_{i}",
                f"Fork {i}",
                capacity=1,
                resource_type=ResourceType.TOOL
            )
            fork_place.add_token(ColoredToken(f"fork_token_{i}", TokenColor.RED))
            self.petri_net.add_place(fork_place)
        
        # Create transitions for philosopher actions
        for i in range(self.num_philosophers):
            left_fork = i
            right_fork = (i + 1) % self.num_philosophers
            
            # Transition: Think -> Hungry
            think_to_hungry = Transition(
                f"think_to_hungry_{i}",
                f"Philosopher {i} Gets Hungry",
                input_places=[f"thinking_{i}"],
                output_places=[f"hungry_{i}"],
                priority=1
            )
            self.petri_net.add_transition(think_to_hungry)
            self.petri_net.add_arc(f"thinking_{i}", f"think_to_hungry_{i}")
            self.petri_net.add_arc(f"think_to_hungry_{i}", f"hungry_{i}")
            
            # Transition: Hungry -> Eating (acquire both forks)
            hungry_to_eating = Transition(
                f"hungry_to_eating_{i}",
                f"Philosopher {i} Starts Eating",
                input_places=[f"hungry_{i}", f"fork_{left_fork}", f"fork_{right_fork}"],
                output_places=[f"eating_{i}"],
                priority=2,
                guard_function=lambda places, i=i, left=left_fork, right=right_fork: 
                    self._can_acquire_forks(places, i, left, right)
            )
            self.petri_net.add_transition(hungry_to_eating)
            self.petri_net.add_arc(f"hungry_{i}", f"hungry_to_eating_{i}")
            self.petri_net.add_arc(f"fork_{left_fork}", f"hungry_to_eating_{i}")
            self.petri_net.add_arc(f"fork_{right_fork}", f"hungry_to_eating_{i}")
            self.petri_net.add_arc(f"hungry_to_eating_{i}", f"eating_{i}")
            
            # Transition: Eating -> Thinking (release both forks)
            eating_to_think = Transition(
                f"eating_to_think_{i}",
                f"Philosopher {i} Finishes Eating",
                input_places=[f"eating_{i}"],
                output_places=[f"thinking_{i}", f"fork_{left_fork}", f"fork_{right_fork}"],
                priority=1
            )
            self.petri_net.add_transition(eating_to_think)
            self.petri_net.add_arc(f"eating_{i}", f"eating_to_think_{i}")
            self.petri_net.add_arc(f"eating_to_think_{i}", f"thinking_{i}")
            self.petri_net.add_arc(f"eating_to_think_{i}", f"fork_{left_fork}")
            self.petri_net.add_arc(f"eating_to_think_{i}", f"fork_{right_fork}")
    
    def _can_acquire_forks(self, places: Dict[str, Place], philosopher_id: int, 
                          left_fork: int, right_fork: int) -> bool:
        """Check if philosopher can acquire both forks"""
        left_fork_place = places.get(f"fork_{left_fork}")
        right_fork_place = places.get(f"fork_{right_fork}")
        
        if not left_fork_place or not right_fork_place:
            return False
        
        # Check if both forks are available
        left_available = len(left_fork_place.tokens) > 0
        right_available = len(right_fork_place.tokens) > 0
        
        return left_available and right_available
    
    def detect_deadlock_neural(self) -> Tuple[bool, float, str]:
        """Detect deadlock using neural network"""
        # Convert Petri net state to input vector
        state_vector = self._get_state_vector()
        
        # Use neural network to detect deadlock
        is_deadlock, confidence = self.neural_controller.detect_deadlock(state_vector)
        
        # Suggest control action
        action = self.neural_controller.suggest_control_action(state_vector)
        
        return is_deadlock, confidence, action
    
    def _get_state_vector(self) -> np.ndarray:
        """Convert Petri net state to vector for neural network"""
        state_vector = []
        
        for i in range(self.num_philosophers):
            # Add philosopher states (thinking=1, hungry=2, eating=3)
            thinking_tokens = len(self.petri_net.places[f"thinking_{i}"].tokens)
            hungry_tokens = len(self.petri_net.places[f"hungry_{i}"].tokens)
            eating_tokens = len(self.petri_net.places[f"eating_{i}"].tokens)
            
            if thinking_tokens > 0:
                state_vector.extend([1, 0, 0])
            elif hungry_tokens > 0:
                state_vector.extend([0, 1, 0])
            elif eating_tokens > 0:
                state_vector.extend([0, 0, 1])
            else:
                state_vector.extend([0, 0, 0])
        
        return np.array(state_vector)
    
    def detect_faults(self) -> Tuple[bool, List[FaultType], Dict[str, Any]]:
        """Detect faults in the philosophers system"""
        system_state = {
            'timestamp': time.time(),
            'philosophers_count': self.num_philosophers,
            'active_transitions': len([t for t in self.petri_net.transitions.values() if t.can_fire(self.petri_net.places)])
        }
        
        return self.fault_detector.detect_fault(self.petri_net, system_state)
    
    def apply_control_action(self, action: str, philosopher_id: Optional[int] = None) -> bool:
        """Apply control action to prevent/resolve deadlock"""
        if action == "immediate_intervention":
            # Force release one philosopher's forks
            if philosopher_id is None:
                philosopher_id = random.randint(0, self.num_philosophers - 1)
            
            # Force transition from eating to thinking
            eating_place = self.petri_net.places[f"eating_{philosopher_id}"]
            if len(eating_place.tokens) > 0:
                # Remove eating token
                eating_place.tokens.pop(0)
                
                # Add thinking token
                thinking_place = self.petri_net.places[f"thinking_{philosopher_id}"]
                thinking_place.add_token(ColoredToken(f"control_token_{philosopher_id}", TokenColor.GREEN))
                
                # Release forks
                left_fork = philosopher_id
                right_fork = (philosopher_id + 1) % self.num_philosophers
                
                self.petri_net.places[f"fork_{left_fork}"].add_token(
                    ColoredToken(f"released_fork_{left_fork}", TokenColor.RED)
                )
                self.petri_net.places[f"fork_{right_fork}"].add_token(
                    ColoredToken(f"released_fork_{right_fork}", TokenColor.RED)
                )
                
                self.control_actions.append({
                    'timestamp': time.time(),
                    'action': action,
                    'philosopher_id': philosopher_id,
                    'success': True
                })
                
                return True
        
        elif action == "monitor_closely":
            # Just monitor, no action needed
            self.control_actions.append({
                'timestamp': time.time(),
                'action': action,
                'philosopher_id': philosopher_id,
                'success': True
            })
            return True
        
        elif action == "preventive_action":
            # Prevent philosophers from getting hungry
            for i in range(self.num_philosophers):
                hungry_place = self.petri_net.places[f"hungry_{i}"]
                if len(hungry_place.tokens) > 0:
                    # Move back to thinking
                    hungry_place.tokens.pop(0)
                    thinking_place = self.petri_net.places[f"thinking_{i}"]
                    thinking_place.add_token(ColoredToken(f"preventive_token_{i}", TokenColor.YELLOW))
            
            self.control_actions.append({
                'timestamp': time.time(),
                'action': action,
                'philosopher_id': philosopher_id,
                'success': True
            })
            return True
        
        return False
    
    def simulate_philosophers(self, duration: float = 60.0) -> Dict[str, Any]:
        """Simulate dining philosophers with neural network control"""
        print(f"Starting philosophers simulation with neural network control...")
        
        start_time = time.time()
        simulation_steps = 0
        deadlock_detections = 0
        fault_detections = 0
        successful_controls = 0
        
        while time.time() - start_time < duration:
            # Detect deadlock using neural network
            is_deadlock, confidence, action = self.detect_deadlock_neural()
            
            if is_deadlock and confidence > 0.7:
                deadlock_detections += 1
                print(f"Deadlock detected with confidence: {confidence:.3f}, Action: {action}")
                
                # Apply control action
                if self.apply_control_action(action):
                    successful_controls += 1
                    print(f"Control action '{action}' applied successfully")
                
                self.deadlock_events.append({
                    'timestamp': time.time(),
                    'confidence': confidence,
                    'action': action,
                    'success': True
                })
            
            # Detect faults
            is_fault, faults, fault_info = self.detect_faults()
            
            if is_fault:
                fault_detections += 1
                print(f"Faults detected: {[f.value for f in faults]}")
                
                # Treat faults
                for fault_type in faults:
                    success, treatment = self.fault_detector.treat_fault(fault_type, fault_info[fault_type])
                    if success:
                        print(f"Successfully treated {fault_type.value} using {treatment}")
                    else:
                        print(f"Failed to treat {fault_type.value}")
                
                self.fault_events.append({
                    'timestamp': time.time(),
                    'faults': [f.value for f in faults],
                    'fault_info': fault_info
                })
            
            # Run Petri net simulation step
            if self.petri_net.simulate_step():
                simulation_steps += 1
            
            time.sleep(0.01)  # Small delay for demonstration
        
        simulation_time = time.time() - start_time
        
        # Calculate performance metrics
        throughput = simulation_steps / simulation_time
        deadlock_rate = deadlock_detections / simulation_steps if simulation_steps > 0 else 0
        fault_rate = fault_detections / simulation_steps if simulation_steps > 0 else 0
        control_success_rate = successful_controls / deadlock_detections if deadlock_detections > 0 else 0
        
        performance_metrics = {
            'simulation_time': simulation_time,
            'simulation_steps': simulation_steps,
            'throughput': throughput,
            'deadlock_detections': deadlock_detections,
            'deadlock_rate': deadlock_rate,
            'fault_detections': fault_detections,
            'fault_rate': fault_rate,
            'successful_controls': successful_controls,
            'control_success_rate': control_success_rate,
            'total_philosophers': self.num_philosophers
        }
        
        print(f"\nSimulation completed!")
        print(f"Total steps: {simulation_steps}")
        print(f"Deadlocks detected: {deadlock_detections}")
        print(f"Faults detected: {fault_detections}")
        print(f"Successful controls: {successful_controls}")
        
        return performance_metrics
    
    def train_neural_controller(self, epochs: int = 1000):
        """Train the neural network controller"""
        print("Training neural network controller...")
        
        # Generate training data
        X_train = []
        y_train = []
        
        for _ in range(500):
            # Generate random philosopher states
            state_vector = np.random.randint(0, 2, self.num_philosophers * 3)
            X_train.append(state_vector)
            
            # Generate deadlock label (simplified)
            # Deadlock if many philosophers are hungry
            hungry_count = np.sum(state_vector[1::3])  # Count hungry philosophers
            is_deadlock = 1 if hungry_count > self.num_philosophers * 0.8 else 0
            y_train.append([is_deadlock])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Train neural network
        self.neural_controller.train(X_train, y_train, epochs=epochs, learning_rate=0.1)
        
        print(f"Neural network training completed!")
        print(f"Final accuracy: {self.neural_controller.accuracy_history[-1]:.4f}")


class PhilosophersAccuracyEvaluator:
    """
    Accuracy evaluator for philosophers deadlock detection
    """
    
    def __init__(self, num_philosophers: int = 100):
        self.num_philosophers = num_philosophers
        self.test_results: List[Dict[str, Any]] = []
        
    def create_test_scenarios(self) -> List[Dict[str, Any]]:
        """Create test scenarios for accuracy evaluation"""
        scenarios = []
        
        # Scenario 1: No deadlock - few philosophers hungry
        scenarios.append({
            'name': 'No Deadlock - Few Hungry',
            'description': 'Only a few philosophers are hungry',
            'expected_deadlock': False,
            'hungry_count': 5,
            'eating_count': 10
        })
        
        # Scenario 2: Deadlock - all philosophers hungry
        scenarios.append({
            'name': 'Deadlock - All Hungry',
            'description': 'All philosophers are hungry (circular wait)',
            'expected_deadlock': True,
            'hungry_count': self.num_philosophers,
            'eating_count': 0
        })
        
        # Scenario 3: Partial deadlock - many hungry
        scenarios.append({
            'name': 'Partial Deadlock - Many Hungry',
            'description': 'Most philosophers are hungry',
            'expected_deadlock': True,
            'hungry_count': int(self.num_philosophers * 0.8),
            'eating_count': 5
        })
        
        # Scenario 4: Mixed state
        scenarios.append({
            'name': 'Mixed State',
            'description': 'Mixed philosopher states',
            'expected_deadlock': False,
            'hungry_count': 20,
            'eating_count': 30,
            'thinking_count': 50
        })
        
        return scenarios
    
    def evaluate_scenario(self, scenario: Dict[str, Any], philosophers_system: PhilosophersPetriNet) -> Dict[str, Any]:
        """Evaluate a single scenario"""
        print(f"\nEvaluating scenario: {scenario['name']}")
        
        # Setup scenario state
        self._setup_scenario_state(philosophers_system, scenario)
        
        # Detect deadlock
        is_deadlock, confidence, action = philosophers_system.detect_deadlock_neural()
        
        # Determine if detection was correct
        expected_deadlock = scenario['expected_deadlock']
        
        if expected_deadlock and is_deadlock:
            result_type = "True Positive"
        elif expected_deadlock and not is_deadlock:
            result_type = "False Negative"
        elif not expected_deadlock and is_deadlock:
            result_type = "False Positive"
        else:
            result_type = "True Negative"
        
        result = {
            'scenario_name': scenario['name'],
            'expected_deadlock': expected_deadlock,
            'detected_deadlock': is_deadlock,
            'confidence': confidence,
            'action': action,
            'result_type': result_type,
            'timestamp': time.time()
        }
        
        print(f"Result: {result_type}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Action: {action}")
        
        return result
    
    def _setup_scenario_state(self, philosophers_system: PhilosophersPetriNet, scenario: Dict[str, Any]):
        """Setup Petri net state for scenario"""
        # Clear all tokens
        for place in philosophers_system.petri_net.places.values():
            place.tokens.clear()
        
        # Add tokens based on scenario
        hungry_count = scenario.get('hungry_count', 0)
        eating_count = scenario.get('eating_count', 0)
        thinking_count = scenario.get('thinking_count', self.num_philosophers - hungry_count - eating_count)
        
        # Add thinking philosophers
        for i in range(min(thinking_count, self.num_philosophers)):
            thinking_place = philosophers_system.petri_net.places[f"thinking_{i}"]
            thinking_place.add_token(ColoredToken(f"think_token_{i}", TokenColor.BLUE))
        
        # Add hungry philosophers
        for i in range(thinking_count, min(thinking_count + hungry_count, self.num_philosophers)):
            hungry_place = philosophers_system.petri_net.places[f"hungry_{i}"]
            hungry_place.add_token(ColoredToken(f"hungry_token_{i}", TokenColor.ORANGE))
        
        # Add eating philosophers
        for i in range(thinking_count + hungry_count, 
                      min(thinking_count + hungry_count + eating_count, self.num_philosophers)):
            eating_place = philosophers_system.petri_net.places[f"eating_{i}"]
            eating_place.add_token(ColoredToken(f"eating_token_{i}", TokenColor.GREEN))
        
        # Add fork tokens
        for i in range(self.num_philosophers):
            fork_place = philosophers_system.petri_net.places[f"fork_{i}"]
            fork_place.add_token(ColoredToken(f"fork_token_{i}", TokenColor.RED))
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive accuracy evaluation"""
        print("=" * 80)
        print("COMPREHENSIVE ACCURACY EVALUATION")
        print("Kaid et al. (2021) Methodology - 100 Philosophers")
        print("=" * 80)
        
        # Create philosophers system
        philosophers_system = PhilosophersPetriNet(self.num_philosophers)
        
        # Train neural network
        philosophers_system.train_neural_controller(epochs=500)
        
        # Create test scenarios
        scenarios = self.create_test_scenarios()
        
        # Evaluate each scenario
        for scenario in scenarios:
            result = self.evaluate_scenario(scenario, philosophers_system)
            self.test_results.append(result)
        
        # Calculate accuracy metrics
        accuracy_metrics = self._calculate_accuracy_metrics()
        
        # Print summary
        self._print_accuracy_summary(accuracy_metrics)
        
        return {
            'accuracy_metrics': accuracy_metrics,
            'test_results': self.test_results,
            'total_tests': len(self.test_results)
        }
    
    def _calculate_accuracy_metrics(self) -> Dict[str, float]:
        """Calculate accuracy metrics"""
        true_positives = sum(1 for r in self.test_results if r['result_type'] == 'True Positive')
        false_positives = sum(1 for r in self.test_results if r['result_type'] == 'False Positive')
        true_negatives = sum(1 for r in self.test_results if r['result_type'] == 'True Negative')
        false_negatives = sum(1 for r in self.test_results if r['result_type'] == 'False Negative')
        
        total = len(self.test_results)
        
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'specificity': specificity,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }
    
    def _print_accuracy_summary(self, metrics: Dict[str, float]):
        """Print accuracy summary"""
        print("\n" + "=" * 80)
        print("ACCURACY EVALUATION SUMMARY")
        print("=" * 80)
        
        print(f"\nOverall Performance:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"  Recall: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"  F1-Score: {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        print(f"  Specificity: {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
        
        print(f"\nConfusion Matrix:")
        print(f"  True Positives: {metrics['true_positives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  True Negatives: {metrics['true_negatives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        
        print(f"\nDetailed Results:")
        for i, result in enumerate(self.test_results, 1):
            print(f"  {i}. {result['scenario_name']}: {result['result_type']} (confidence: {result['confidence']:.3f})")


def run_philosophers_kaid_experiment():
    """Run comprehensive experiment using Kaid et al. methodology"""
    print("=" * 80)
    print("KAID ET AL. (2021) METHODOLOGY - 100 PHILOSOPHERS EXPERIMENT")
    print("Colored Resource-Oriented Petri Nets with Neural Network Control")
    print("=" * 80)
    
    # Run accuracy evaluation
    evaluator = PhilosophersAccuracyEvaluator(num_philosophers=100)
    evaluation_results = evaluator.run_comprehensive_evaluation()
    
    # Run simulation
    print("\nRunning philosophers simulation...")
    philosophers_system = PhilosophersPetriNet(num_philosophers=100)
    philosophers_system.train_neural_controller(epochs=500)
    
    simulation_results = philosophers_system.simulate_philosophers(duration=30.0)
    
    # Print final results
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED!")
    print("=" * 80)
    
    print(f"\nAccuracy Results:")
    metrics = evaluation_results['accuracy_metrics']
    print(f"  Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    
    print(f"\nSimulation Results:")
    print(f"  Total Steps: {simulation_results['simulation_steps']}")
    print(f"  Deadlock Detections: {simulation_results['deadlock_detections']}")
    print(f"  Fault Detections: {simulation_results['fault_detections']}")
    print(f"  Control Success Rate: {simulation_results['control_success_rate']:.4f}")
    
    return evaluation_results, simulation_results


if __name__ == "__main__":
    eval_results, sim_results = run_philosophers_kaid_experiment()
