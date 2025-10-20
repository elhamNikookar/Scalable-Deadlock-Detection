#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SDD (Scalable Deadlock Detection) Approach for Classic Concurrency Benchmarks

This implementation covers all major concurrency problems:
- DPH – Dining Philosophers
- SHP – Sleeping Barber
- PLC – Producer–Consumer
- TA – Train Allocation
- FIR – Cigarette Smokers
- RSC – Rail Safety Controller
- BRP – Bridge Crossing Problem
- BTS – Bank Transfer System
- ATSV – Elevator System
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
import threading
import queue

class BenchmarkType(Enum):
    """Types of concurrency benchmarks"""
    DPH = "dining_philosophers"
    SHP = "sleeping_barber"
    PLC = "producer_consumer"
    TA = "train_allocation"
    FIR = "cigarette_smokers"
    RSC = "rail_safety_controller"
    BRP = "bridge_crossing"
    BTS = "bank_transfer"
    ATSV = "elevator_system"

@dataclass
class Process:
    """Generic process representation"""
    id: int
    state: str = "idle"
    resources: Set[int] = None
    waiting_for: Set[int] = None
    priority: int = 1
    execution_time: float = 0.0
    deadlock_risk: float = 0.0
    benchmark_type: BenchmarkType = None
    
    def __post_init__(self):
        if self.resources is None:
            self.resources = set()
        if self.waiting_for is None:
            self.waiting_for = set()

@dataclass
class Resource:
    """Generic resource representation"""
    id: int
    available: bool = True
    owner: Optional[int] = None
    request_queue: List[int] = None
    type: str = "shared"
    contention_level: float = 0.0
    capacity: int = 1
    
    def __post_init__(self):
        if self.request_queue is None:
            self.request_queue = []

class SDDBenchmark:
    """SDD implementation for all concurrency benchmarks"""
    
    def __init__(self, benchmark_type: BenchmarkType, num_processes: int = 50):
        self.benchmark_type = benchmark_type
        self.num_processes = num_processes
        self.processes = []
        self.resources = []
        self.graph = nx.DiGraph()
        self.deadlock_history = []
        self.performance_metrics = {}
        
        # Initialize based on benchmark type
        self._initialize_benchmark()
    
    def _initialize_benchmark(self):
        """Initialize the specific benchmark"""
        if self.benchmark_type == BenchmarkType.DPH:
            self._init_dining_philosophers()
        elif self.benchmark_type == BenchmarkType.SHP:
            self._init_sleeping_barber()
        elif self.benchmark_type == BenchmarkType.PLC:
            self._init_producer_consumer()
        elif self.benchmark_type == BenchmarkType.TA:
            self._init_train_allocation()
        elif self.benchmark_type == BenchmarkType.FIR:
            self._init_cigarette_smokers()
        elif self.benchmark_type == BenchmarkType.RSC:
            self._init_rail_safety_controller()
        elif self.benchmark_type == BenchmarkType.BRP:
            self._init_bridge_crossing()
        elif self.benchmark_type == BenchmarkType.BTS:
            self._init_bank_transfer()
        elif self.benchmark_type == BenchmarkType.ATSV:
            self._init_elevator_system()
    
    def _init_dining_philosophers(self):
        """Initialize Dining Philosophers Problem"""
        # Create philosophers
        for i in range(self.num_processes):
            process = Process(
                id=i,
                state="thinking",
                benchmark_type=BenchmarkType.DPH
            )
            self.processes.append(process)
            self.graph.add_node(f"philosopher_{i}", type="philosopher", state="thinking")
        
        # Create forks
        for i in range(self.num_processes):
            resource = Resource(id=i, type="fork")
            self.resources.append(resource)
            self.graph.add_node(f"fork_{i}", type="fork", available=True)
        
        # Add dependency edges
        for i in range(self.num_processes):
            left_fork = i
            right_fork = (i + 1) % self.num_processes
            self.graph.add_edge(f"philosopher_{i}", f"fork_{left_fork}", type="needs_left")
            self.graph.add_edge(f"philosopher_{i}", f"fork_{right_fork}", type="needs_right")
    
    def _init_sleeping_barber(self):
        """Initialize Sleeping Barber Problem"""
        # Create barber
        barber = Process(id=0, state="sleeping", benchmark_type=BenchmarkType.SHP)
        self.processes.append(barber)
        self.graph.add_node("barber", type="barber", state="sleeping")
        
        # Create customers
        for i in range(1, self.num_processes):
            customer = Process(id=i, state="waiting", benchmark_type=BenchmarkType.SHP)
            self.processes.append(customer)
            self.graph.add_node(f"customer_{i}", type="customer", state="waiting")
        
        # Create barber chair
        chair = Resource(id=0, type="chair", capacity=1)
        self.resources.append(chair)
        self.graph.add_node("chair", type="chair", available=True)
        
        # Create waiting room
        waiting_room = Resource(id=1, type="waiting_room", capacity=self.num_processes-1)
        self.resources.append(waiting_room)
        self.graph.add_node("waiting_room", type="waiting_room", available=True)
    
    def _init_producer_consumer(self):
        """Initialize Producer-Consumer Problem"""
        # Create producers
        num_producers = self.num_processes // 2
        for i in range(num_producers):
            producer = Process(id=i, state="producing", benchmark_type=BenchmarkType.PLC)
            self.processes.append(producer)
            self.graph.add_node(f"producer_{i}", type="producer", state="producing")
        
        # Create consumers
        for i in range(num_producers, self.num_processes):
            consumer = Process(id=i, state="consuming", benchmark_type=BenchmarkType.PLC)
            self.processes.append(consumer)
            self.graph.add_node(f"consumer_{i}", type="consumer", state="consuming")
        
        # Create buffer
        buffer = Resource(id=0, type="buffer", capacity=10)
        self.resources.append(buffer)
        self.graph.add_node("buffer", type="buffer", available=True)
        
        # Create mutex
        mutex = Resource(id=1, type="mutex", capacity=1)
        self.resources.append(mutex)
        self.graph.add_node("mutex", type="mutex", available=True)
    
    def _init_train_allocation(self):
        """Initialize Train Allocation Problem"""
        # Create trains
        for i in range(self.num_processes):
            train = Process(id=i, state="waiting", benchmark_type=BenchmarkType.TA)
            self.processes.append(train)
            self.graph.add_node(f"train_{i}", type="train", state="waiting")
        
        # Create tracks
        num_tracks = self.num_processes // 2
        for i in range(num_tracks):
            track = Resource(id=i, type="track", capacity=1)
            self.resources.append(track)
            self.graph.add_node(f"track_{i}", type="track", available=True)
        
        # Create signals
        for i in range(num_tracks):
            signal = Resource(id=i+num_tracks, type="signal", capacity=1)
            self.resources.append(signal)
            self.graph.add_node(f"signal_{i}", type="signal", available=True)
    
    def _init_cigarette_smokers(self):
        """Initialize Cigarette Smokers Problem"""
        # Create smokers
        for i in range(3):  # Only 3 smokers
            smoker = Process(id=i, state="waiting", benchmark_type=BenchmarkType.FIR)
            self.processes.append(smoker)
            self.graph.add_node(f"smoker_{i}", type="smoker", state="waiting")
        
        # Create agent
        agent = Process(id=3, state="working", benchmark_type=BenchmarkType.FIR)
        self.processes.append(agent)
        self.graph.add_node("agent", type="agent", state="working")
        
        # Create ingredients
        ingredients = ["tobacco", "paper", "matches"]
        for i, ingredient in enumerate(ingredients):
            resource = Resource(id=i, type=ingredient, capacity=1)
            self.resources.append(resource)
            self.graph.add_node(f"ingredient_{i}", type=ingredient, available=True)
    
    def _init_rail_safety_controller(self):
        """Initialize Rail Safety Controller Problem"""
        # Create trains
        for i in range(self.num_processes):
            train = Process(id=i, state="waiting", benchmark_type=BenchmarkType.RSC)
            self.processes.append(train)
            self.graph.add_node(f"train_{i}", type="train", state="waiting")
        
        # Create sections
        num_sections = self.num_processes // 3
        for i in range(num_sections):
            section = Resource(id=i, type="section", capacity=1)
            self.resources.append(section)
            self.graph.add_node(f"section_{i}", type="section", available=True)
        
        # Create signals
        for i in range(num_sections):
            signal = Resource(id=i+num_sections, type="signal", capacity=1)
            self.resources.append(signal)
            self.graph.add_node(f"signal_{i}", type="signal", available=True)
    
    def _init_bridge_crossing(self):
        """Initialize Bridge Crossing Problem"""
        # Create people
        for i in range(self.num_processes):
            person = Process(id=i, state="waiting", benchmark_type=BenchmarkType.BRP)
            self.processes.append(person)
            self.graph.add_node(f"person_{i}", type="person", state="waiting")
        
        # Create bridge
        bridge = Resource(id=0, type="bridge", capacity=1)
        self.resources.append(bridge)
        self.graph.add_node("bridge", type="bridge", available=True)
        
        # Create mutex
        mutex = Resource(id=1, type="mutex", capacity=1)
        self.resources.append(mutex)
        self.graph.add_node("mutex", type="mutex", available=True)
    
    def _init_bank_transfer(self):
        """Initialize Bank Transfer System"""
        # Create accounts
        for i in range(self.num_processes):
            account = Process(id=i, state="active", benchmark_type=BenchmarkType.BTS)
            self.processes.append(account)
            self.graph.add_node(f"account_{i}", type="account", state="active")
        
        # Create locks
        for i in range(self.num_processes):
            lock = Resource(id=i, type="lock", capacity=1)
            self.resources.append(lock)
            self.graph.add_node(f"lock_{i}", type="lock", available=True)
    
    def _init_elevator_system(self):
        """Initialize Elevator System"""
        # Create elevators
        num_elevators = min(3, self.num_processes // 10)
        for i in range(num_elevators):
            elevator = Process(id=i, state="idle", benchmark_type=BenchmarkType.ATSV)
            self.processes.append(elevator)
            self.graph.add_node(f"elevator_{i}", type="elevator", state="idle")
        
        # Create floors
        num_floors = min(10, self.num_processes // 5)
        for i in range(num_floors):
            floor = Resource(id=i, type="floor", capacity=1)
            self.resources.append(floor)
            self.graph.add_node(f"floor_{i}", type="floor", available=True)
        
        # Create call buttons
        for i in range(num_floors):
            button = Resource(id=i+num_floors, type="button", capacity=1)
            self.resources.append(button)
            self.graph.add_node(f"button_{i}", type="button", available=True)
    
    def _detect_deadlock_traditional(self) -> Tuple[bool, Dict]:
        """Traditional deadlock detection using cycle detection"""
        try:
            # Find strongly connected components
            sccs = list(nx.strongly_connected_components(self.graph))
            deadlock_info = {
                'type': 'none',
                'cycles': [],
                'severity': 0.0,
                'affected_processes': [],
                'benchmark_type': self.benchmark_type.value
            }
            
            for scc in sccs:
                if len(scc) > 1:
                    # Check if this is a deadlock cycle
                    process_nodes = [n for n in scc if n.startswith(('philosopher_', 'barber', 'customer_', 'producer_', 'consumer_', 'train_', 'smoker_', 'person_', 'account_', 'elevator_'))]
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
            return False, {'type': 'none', 'cycles': [], 'severity': 0.0, 'affected_processes': [], 'benchmark_type': self.benchmark_type.value}
    
    def _simulate_benchmark_evolution(self, num_steps: int = 100):
        """Simulate benchmark-specific system evolution"""
        for step in range(num_steps):
            if self.benchmark_type == BenchmarkType.DPH:
                self._simulate_dining_philosophers()
            elif self.benchmark_type == BenchmarkType.SHP:
                self._simulate_sleeping_barber()
            elif self.benchmark_type == BenchmarkType.PLC:
                self._simulate_producer_consumer()
            elif self.benchmark_type == BenchmarkType.TA:
                self._simulate_train_allocation()
            elif self.benchmark_type == BenchmarkType.FIR:
                self._simulate_cigarette_smokers()
            elif self.benchmark_type == BenchmarkType.RSC:
                self._simulate_rail_safety_controller()
            elif self.benchmark_type == BenchmarkType.BRP:
                self._simulate_bridge_crossing()
            elif self.benchmark_type == BenchmarkType.BTS:
                self._simulate_bank_transfer()
            elif self.benchmark_type == BenchmarkType.ATSV:
                self._simulate_elevator_system()
    
    def _simulate_dining_philosophers(self):
        """Simulate Dining Philosophers evolution"""
        philosopher_id = random.randint(0, len(self.processes) - 1)
        philosopher = self.processes[philosopher_id]
        
        if philosopher.state == "thinking" and random.random() < 0.3:
            philosopher.state = "hungry"
            self.graph.nodes[f"philosopher_{philosopher_id}"]['state'] = "hungry"
        elif philosopher.state == "hungry":
            left_fork = philosopher_id
            right_fork = (philosopher_id + 1) % len(self.processes)
            left_resource = self.resources[left_fork]
            right_resource = self.resources[right_fork]
            
            if left_resource.available and right_resource.available:
                left_resource.available = False
                right_resource.available = False
                left_resource.owner = philosopher_id
                right_resource.owner = philosopher_id
                philosopher.state = "eating"
                philosopher.resources.add(left_fork)
                philosopher.resources.add(right_fork)
                self.graph.nodes[f"philosopher_{philosopher_id}"]['state'] = "eating"
        elif philosopher.state == "eating" and random.random() < 0.4:
            for resource_id in list(philosopher.resources):
                resource = self.resources[resource_id]
                resource.available = True
                resource.owner = None
            philosopher.resources.clear()
            philosopher.state = "thinking"
            self.graph.nodes[f"philosopher_{philosopher_id}"]['state'] = "thinking"
    
    def _simulate_sleeping_barber(self):
        """Simulate Sleeping Barber evolution"""
        barber = self.processes[0]
        chair = self.resources[0]
        waiting_room = self.resources[1]
        
        if barber.state == "sleeping" and waiting_room.request_queue:
            # Barber wakes up
            barber.state = "working"
            self.graph.nodes["barber"]['state'] = "working"
        elif barber.state == "working" and chair.available:
            # Barber takes customer
            if waiting_room.request_queue:
                customer_id = waiting_room.request_queue.pop(0)
                chair.available = False
                chair.owner = customer_id
                self.processes[customer_id].state = "getting_haircut"
                self.graph.nodes[f"customer_{customer_id}"]['state'] = "getting_haircut"
        elif barber.state == "working" and not chair.available:
            # Barber finishes haircut
            chair.available = True
            chair.owner = None
            barber.state = "sleeping"
            self.graph.nodes["barber"]['state'] = "sleeping"
    
    def _simulate_producer_consumer(self):
        """Simulate Producer-Consumer evolution"""
        # Randomly select producer or consumer
        if random.random() < 0.5:
            # Producer
            producer_id = random.randint(0, len(self.processes) // 2 - 1)
            producer = self.processes[producer_id]
            buffer = self.resources[0]
            mutex = self.resources[1]
            
            if producer.state == "producing" and mutex.available:
                mutex.available = False
                mutex.owner = producer_id
                producer.state = "writing"
                self.graph.nodes[f"producer_{producer_id}"]['state'] = "writing"
        else:
            # Consumer
            consumer_id = random.randint(len(self.processes) // 2, len(self.processes) - 1)
            consumer = self.processes[consumer_id]
            buffer = self.resources[0]
            mutex = self.resources[1]
            
            if consumer.state == "consuming" and mutex.available:
                mutex.available = False
                mutex.owner = consumer_id
                consumer.state = "reading"
                self.graph.nodes[f"consumer_{consumer_id}"]['state'] = "reading"
    
    def _simulate_train_allocation(self):
        """Simulate Train Allocation evolution"""
        train_id = random.randint(0, len(self.processes) - 1)
        train = self.processes[train_id]
        
        if train.state == "waiting":
            # Try to allocate track
            track_id = random.randint(0, len(self.resources) // 2 - 1)
            track = self.resources[track_id]
            
            if track.available:
                track.available = False
                track.owner = train_id
                train.state = "running"
                self.graph.nodes[f"train_{train_id}"]['state'] = "running"
        elif train.state == "running" and random.random() < 0.3:
            # Release track
            for resource in self.resources:
                if resource.owner == train_id:
                    resource.available = True
                    resource.owner = None
            train.state = "waiting"
            self.graph.nodes[f"train_{train_id}"]['state'] = "waiting"
    
    def _simulate_cigarette_smokers(self):
        """Simulate Cigarette Smokers evolution"""
        agent = self.processes[3]
        
        if agent.state == "working":
            # Agent provides ingredients
            ingredient_id = random.randint(0, 2)
            ingredient = self.resources[ingredient_id]
            ingredient.available = True
            agent.state = "waiting"
            self.graph.nodes["agent"]['state'] = "waiting"
        elif agent.state == "waiting":
            # Check if smoker can smoke
            for smoker_id in range(3):
                smoker = self.processes[smoker_id]
                if smoker.state == "waiting":
                    # Check if smoker has ingredients
                    has_ingredients = all(self.resources[i].available for i in range(3) if i != smoker_id)
                    if has_ingredients:
                        smoker.state = "smoking"
                        self.graph.nodes[f"smoker_{smoker_id}"]['state'] = "smoking"
                        agent.state = "working"
                        self.graph.nodes["agent"]['state'] = "working"
                        break
    
    def _simulate_rail_safety_controller(self):
        """Simulate Rail Safety Controller evolution"""
        train_id = random.randint(0, len(self.processes) - 1)
        train = self.processes[train_id]
        
        if train.state == "waiting":
            # Try to enter section
            section_id = random.randint(0, len(self.resources) // 2 - 1)
            section = self.resources[section_id]
            
            if section.available:
                section.available = False
                section.owner = train_id
                train.state = "running"
                self.graph.nodes[f"train_{train_id}"]['state'] = "running"
        elif train.state == "running" and random.random() < 0.3:
            # Leave section
            for resource in self.resources:
                if resource.owner == train_id:
                    resource.available = True
                    resource.owner = None
            train.state = "waiting"
            self.graph.nodes[f"train_{train_id}"]['state'] = "waiting"
    
    def _simulate_bridge_crossing(self):
        """Simulate Bridge Crossing evolution"""
        person_id = random.randint(0, len(self.processes) - 1)
        person = self.processes[person_id]
        bridge = self.resources[0]
        mutex = self.resources[1]
        
        if person.state == "waiting" and bridge.available and mutex.available:
            bridge.available = False
            bridge.owner = person_id
            mutex.available = False
            mutex.owner = person_id
            person.state = "crossing"
            self.graph.nodes[f"person_{person_id}"]['state'] = "crossing"
        elif person.state == "crossing" and random.random() < 0.4:
            bridge.available = True
            bridge.owner = None
            mutex.available = True
            mutex.owner = None
            person.state = "waiting"
            self.graph.nodes[f"person_{person_id}"]['state'] = "waiting"
    
    def _simulate_bank_transfer(self):
        """Simulate Bank Transfer evolution"""
        account_id = random.randint(0, len(self.processes) - 1)
        account = self.processes[account_id]
        lock = self.resources[account_id]
        
        if account.state == "active" and lock.available:
            lock.available = False
            lock.owner = account_id
            account.state = "locked"
            self.graph.nodes[f"account_{account_id}"]['state'] = "locked"
        elif account.state == "locked" and random.random() < 0.3:
            lock.available = True
            lock.owner = None
            account.state = "active"
            self.graph.nodes[f"account_{account_id}"]['state'] = "active"
    
    def _simulate_elevator_system(self):
        """Simulate Elevator System evolution"""
        elevator_id = random.randint(0, min(2, len(self.processes) - 1))
        elevator = self.processes[elevator_id]
        
        if elevator.state == "idle":
            # Try to serve floor
            floor_id = random.randint(0, len(self.resources) // 2 - 1)
            floor = self.resources[floor_id]
            
            if floor.available:
                floor.available = False
                floor.owner = elevator_id
                elevator.state = "moving"
                self.graph.nodes[f"elevator_{elevator_id}"]['state'] = "moving"
        elif elevator.state == "moving" and random.random() < 0.4:
            # Release floor
            for resource in self.resources:
                if resource.owner == elevator_id:
                    resource.available = True
                    resource.owner = None
            elevator.state = "idle"
            self.graph.nodes[f"elevator_{elevator_id}"]['state'] = "idle"
    
    def run_sdd_analysis(self, num_iterations: int = 100) -> Dict:
        """Run SDD analysis for the specific benchmark"""
        print(f"Starting SDD Analysis for {self.benchmark_type.value.upper()}")
        print(f"System: {self.num_processes} processes, {len(self.resources)} resources")
        print("=" * 60)
        
        start_time = time.time()
        results = {
            'benchmark_type': self.benchmark_type.value,
            'traditional_deadlocks': 0,
            'total_iterations': 0,
            'execution_time': 0.0,
            'performance_metrics': {},
            'deadlock_details': []
        }
        
        for iteration in range(num_iterations):
            # Simulate system evolution
            self._simulate_benchmark_evolution(10)
            
            # Traditional deadlock detection
            traditional_deadlock, traditional_info = self._detect_deadlock_traditional()
            
            # Count detections
            if traditional_deadlock:
                results['traditional_deadlocks'] += 1
            
            # Record deadlock details
            if traditional_deadlock:
                deadlock_detail = {
                    'iteration': iteration,
                    'traditional_detected': traditional_deadlock,
                    'traditional_info': traditional_info,
                    'graph_features': self._get_graph_features()
                }
                results['deadlock_details'].append(deadlock_detail)
            
            # Show progress
            if iteration % 20 == 0:
                print(f"Iteration {iteration}/{num_iterations} - "
                      f"Deadlocks: {results['traditional_deadlocks']}")
        
        end_time = time.time()
        results['total_iterations'] = num_iterations
        results['execution_time'] = end_time - start_time
        
        # Calculate performance metrics
        results['performance_metrics'] = {
            'deadlock_detection_rate': results['traditional_deadlocks'] / num_iterations if num_iterations > 0 else 0,
            'avg_deadlock_severity': np.mean([d['traditional_info']['severity'] for d in results['deadlock_details']]) if results['deadlock_details'] else 0,
            'graph_density': self._get_graph_features()['density'],
            'num_cycles': len(results['deadlock_details'])
        }
        
        print(f"\n✅ SDD Analysis completed for {self.benchmark_type.value.upper()}!")
        print(f"⏱️  Execution time: {results['execution_time']:.2f} seconds")
        print(f"🔄 Total iterations: {results['total_iterations']}")
        print(f"🔍 Deadlocks found: {results['traditional_deadlocks']}")
        print(f"📊 Detection rate: {results['performance_metrics']['deadlock_detection_rate']:.3f}")
        
        return results
    
    def _get_graph_features(self) -> Dict:
        """Extract graph features"""
        features = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'avg_degree': np.mean([d for n, d in self.graph.degree()]) if self.graph.number_of_nodes() > 0 else 0,
            'is_strongly_connected': nx.is_strongly_connected(self.graph),
            'num_strongly_connected_components': nx.number_strongly_connected_components(self.graph)
        }
        return features
    
    def visualize_benchmark_state(self, save_path: str = None):
        """Visualize the current benchmark state"""
        plt.figure(figsize=(15, 10))
        
        # Create subplot for graph
        ax1 = plt.subplot(2, 2, 1)
        
        # Color nodes based on type
        node_colors = []
        node_sizes = []
        
        for node in self.graph.nodes():
            if any(node.startswith(prefix) for prefix in ['philosopher_', 'barber', 'customer_', 'producer_', 'consumer_', 'train_', 'smoker_', 'person_', 'account_', 'elevator_']):
                node_colors.append('lightblue')
                node_sizes.append(300)
            else:  # resource
                node_colors.append('lightgreen')
                node_sizes.append(200)
        
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
        
        ax1.set_title(f"{self.benchmark_type.value.upper()} - System Graph")
        
        # Process state distribution
        ax2 = plt.subplot(2, 2, 2)
        states = list(set(p.state for p in self.processes))
        counts = [sum(1 for p in self.processes if p.state == state) for state in states]
        ax2.bar(states, counts, color=['lightblue', 'orange', 'green', 'red'][:len(states)])
        ax2.set_title('Process State Distribution')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
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
        
        features = self._get_graph_features()
        stats_text = f"""
{self.benchmark_type.value.upper()} Statistics:
• Number of processes: {self.num_processes}
• Number of resources: {len(self.resources)}
• Graph nodes: {features['num_nodes']}
• Graph edges: {features['num_edges']}
• Graph density: {features['density']:.3f}
• Strongly connected: {features['is_strongly_connected']}
• SCC components: {features['num_strongly_connected_components']}
        """
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"{self.benchmark_type.value.upper()} visualization saved to {save_path}")
        
        plt.show()
    
    def export_results(self, filename: str = None):
        """Export benchmark results"""
        if filename is None:
            filename = f"sdd_{self.benchmark_type.value}_results.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.performance_metrics, f, indent=2, ensure_ascii=False)
        print(f"{self.benchmark_type.value.upper()} results saved to {filename}")

def run_all_benchmarks(num_iterations: int = 50) -> Dict:
    """Run SDD analysis on all benchmarks"""
    print("SDD (Scalable Deadlock Detection) - All Benchmarks")
    print("=" * 60)
    
    all_results = {}
    
    for benchmark_type in BenchmarkType:
        print(f"\n{'='*20} {benchmark_type.value.upper()} {'='*20}")
        
        # Create benchmark
        benchmark = SDDBenchmark(benchmark_type, num_processes=30)
        
        # Run analysis
        results = benchmark.run_sdd_analysis(num_iterations)
        
        # Store results
        all_results[benchmark_type.value] = results
        
        # Visualize
        benchmark.visualize_benchmark_state(f"sdd_{benchmark_type.value}_visualization.png")
        
        # Export
        benchmark.export_results()
    
    return all_results

def main():
    """Main function for SDD benchmarks"""
    print("SDD (Scalable Deadlock Detection) - Classic Concurrency Benchmarks")
    print("=" * 60)
    
    # Run all benchmarks
    all_results = run_all_benchmarks(num_iterations=50)
    
    # Summary
    print("\n" + "=" * 60)
    print("SDD BENCHMARKS SUMMARY")
    print("=" * 60)
    
    for benchmark_name, results in all_results.items():
        print(f"{benchmark_name.upper():<20} | "
              f"Deadlocks: {results['traditional_deadlocks']:<5} | "
              f"Rate: {results['performance_metrics']['deadlock_detection_rate']:.3f} | "
              f"Time: {results['execution_time']:.2f}s")
    
    print("\n✅ All SDD benchmarks completed successfully!")
    print("📁 Output files:")
    for benchmark_type in BenchmarkType:
        print(f"   - sdd_{benchmark_type.value}_visualization.png")
        print(f"   - sdd_{benchmark_type.value}_results.json")

if __name__ == "__main__":
    main()
