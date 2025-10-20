"""
Graph Transformation System Implementation
Based on Salimi et al. (2020) methodology
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from enum import Enum


class NodeType(Enum):
    """Types of nodes in the graph transformation system"""
    PROCESS = "process"
    RESOURCE = "resource"
    STATE = "state"


@dataclass
class Node:
    """Represents a node in the graph transformation system"""
    id: str
    node_type: NodeType
    properties: Dict[str, any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class Edge:
    """Represents an edge in the graph transformation system"""
    source: str
    target: str
    edge_type: str
    weight: float = 1.0
    properties: Dict[str, any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class GraphTransformationSystem:
    """
    Graph Transformation System for modeling concurrent systems
    Based on the methodology from Salimi et al. (2020)
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.transformation_rules: List[Dict] = []
        self.initial_state: Optional[str] = None
        
    def add_node(self, node: Node):
        """Add a node to the graph transformation system"""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, **node.properties)
        
    def add_edge(self, edge: Edge):
        """Add an edge to the graph transformation system"""
        if edge.source in self.nodes and edge.target in self.nodes:
            self.edges.append(edge)
            self.graph.add_edge(edge.source, edge.target, 
                              weight=edge.weight, 
                              edge_type=edge.edge_type,
                              **edge.properties)
    
    def add_transformation_rule(self, rule: Dict):
        """Add a transformation rule to the system"""
        self.transformation_rules.append(rule)
    
    def get_reachable_states(self, start_state: str, max_depth: int = 10) -> Set[str]:
        """
        Get all reachable states from a given starting state
        This is used for reachability verification
        """
        reachable = set()
        queue = [(start_state, 0)]
        
        while queue:
            current_state, depth = queue.pop(0)
            if depth > max_depth:
                continue
                
            reachable.add(current_state)
            
            # Find all possible next states using transformation rules
            for rule in self.transformation_rules:
                if self._can_apply_rule(current_state, rule):
                    next_state = self._apply_rule(current_state, rule)
                    if next_state not in reachable:
                        queue.append((next_state, depth + 1))
        
        return reachable
    
    def _can_apply_rule(self, state: str, rule: Dict) -> bool:
        """Check if a transformation rule can be applied to a state"""
        # Simplified rule application check
        # In practice, this would involve pattern matching
        return True  # Placeholder implementation
    
    def _apply_rule(self, state: str, rule: Dict) -> str:
        """Apply a transformation rule to a state"""
        # Simplified rule application
        # In practice, this would involve graph rewriting
        return f"{state}_transformed"
    
    def detect_deadlock(self, state: str) -> Tuple[bool, List[str]]:
        """
        Detect deadlock in a given state
        Returns (is_deadlock, deadlock_cycle)
        """
        # Check for circular wait conditions
        try:
            # Find strongly connected components
            scc = list(nx.strongly_connected_components(self.graph))
            
            for component in scc:
                if len(component) > 1:
                    # Check if this component forms a deadlock
                    if self._is_deadlock_cycle(component):
                        return True, list(component)
            
            return False, []
        except:
            return False, []
    
    def _is_deadlock_cycle(self, component: Set[str]) -> bool:
        """Check if a strongly connected component forms a deadlock"""
        # Simplified deadlock detection
        # In practice, this would check for resource allocation patterns
        return len(component) > 2  # Placeholder logic
    
    def get_system_state(self) -> Dict:
        """Get current state of the graph transformation system"""
        return {
            'nodes': len(self.nodes),
            'edges': len(self.edges),
            'rules': len(self.transformation_rules),
            'graph_density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph)
        }


class PetriNetSystem(GraphTransformationSystem):
    """
    Petri Net implementation as a special case of graph transformation system
    """
    
    def __init__(self):
        super().__init__()
        self.places: Dict[str, int] = {}  # Place -> token count
        self.transitions: Set[str] = set()
        
    def add_place(self, place_id: str, initial_tokens: int = 0):
        """Add a place to the Petri net"""
        self.places[place_id] = initial_tokens
        self.add_node(Node(place_id, NodeType.STATE, {'tokens': initial_tokens}))
        
    def add_transition(self, transition_id: str):
        """Add a transition to the Petri net"""
        self.transitions.add(transition_id)
        self.add_node(Node(transition_id, NodeType.PROCESS))
        
    def add_arc(self, source: str, target: str, weight: int = 1):
        """Add an arc between places and transitions"""
        self.add_edge(Edge(source, target, "arc", weight))
        
    def is_transition_enabled(self, transition: str) -> bool:
        """Check if a transition is enabled (can fire)"""
        if transition not in self.transitions:
            return False
            
        # Check input places
        for edge in self.edges:
            if edge.target == transition and edge.source in self.places:
                if self.places[edge.source] < edge.weight:
                    return False
        return True
    
    def fire_transition(self, transition: str) -> bool:
        """Fire a transition if it's enabled"""
        if not self.is_transition_enabled(transition):
            return False
            
        # Consume tokens from input places
        for edge in self.edges:
            if edge.target == transition and edge.source in self.places:
                self.places[edge.source] -= edge.weight
                
        # Produce tokens to output places
        for edge in self.edges:
            if edge.source == transition and edge.target in self.places:
                self.places[edge.target] += edge.weight
                
        return True
    
    def detect_deadlock(self, state: str = None) -> Tuple[bool, List[str]]:
        """Detect deadlock in Petri net"""
        # Check if any transition is enabled
        enabled_transitions = [t for t in self.transitions if self.is_transition_enabled(t)]
        
        if not enabled_transitions:
            # No transitions are enabled - potential deadlock
            return True, []
        
        return False, enabled_transitions
