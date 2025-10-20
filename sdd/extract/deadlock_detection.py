import sqlite3
import pandas as pd
import numpy as np
from dining_philosopher_database import DiningPhilosopherDatabase
from typing import List, Dict, Set, Tuple
import networkx as nx

class DeadlockDetector:
    def __init__(self, db_path: str = "dining_philosopher.db"):
        """Initialize the deadlock detector."""
        self.db = DiningPhilosopherDatabase(db_path)
        
    def detect_deadlocks(self) -> List[Dict]:
        """Detect deadlock states in the dining philosopher system."""
        deadlocks = []
        
        # Get all states
        self.db.cursor.execute('''
            SELECT DISTINCT s.state_id, s.is_start_state, s.is_final_state, s.has_reach_property
            FROM states s
            ORDER BY s.state_id
        ''')
        states = self.db.cursor.fetchall()
        
        for state in states:
            state_id = state[0]
            is_deadlock = self._analyze_state_for_deadlock(state_id)
            
            if is_deadlock:
                deadlock_info = {
                    'state_id': state_id,
                    'is_start_state': state[1],
                    'is_final_state': state[2],
                    'has_reach_property': state[3],
                    'deadlock_type': self._classify_deadlock_type(state_id),
                    'deadlock_reason': self._get_deadlock_reason(state_id),
                    'affected_philosophers': self._get_affected_philosophers(state_id),
                    'resource_usage': self._get_resource_usage(state_id)
                }
                deadlocks.append(deadlock_info)
        
        return deadlocks
    
    def _analyze_state_for_deadlock(self, state_id: str) -> bool:
        """Analyze if a state is a deadlock state."""
        
        # Get all philosophers in this state
        self.db.cursor.execute('''
            SELECT node_id, node_state, left_fork_id, right_fork_id
            FROM state_configurations 
            WHERE state_id = ? AND node_type = 'Philosopher'
        ''', (state_id,))
        philosophers = self.db.cursor.fetchall()
        
        # Get all forks in this state
        self.db.cursor.execute('''
            SELECT node_id, node_state
            FROM state_configurations 
            WHERE state_id = ? AND node_type = 'Fork'
        ''', (state_id,))
        forks = self.db.cursor.fetchall()
        
        # Check for classic dining philosopher deadlock
        if self._is_classic_deadlock(philosophers, forks):
            return True
        
        # Check for resource starvation
        if self._is_resource_starvation(philosophers, forks):
            return True
        
        # Check for circular wait
        if self._is_circular_wait(philosophers):
            return True
        
        # Check for livelock (if applicable)
        if self._is_livelock(state_id):
            return True
        
        return False
    
    def _is_classic_deadlock(self, philosophers: List, forks: List) -> bool:
        """Check for classic dining philosopher deadlock."""
        # Classic deadlock: all philosophers are hungry and each holds one fork
        hungry_philosophers = [p for p in philosophers if p[1] == 'hungry']
        
        if len(hungry_philosophers) == 0:
            return False
        
        # Check if all hungry philosophers hold exactly one fork
        philosophers_with_one_fork = 0
        for phil in hungry_philosophers:
            left_fork = phil[2] is not None
            right_fork = phil[3] is not None
            if (left_fork and not right_fork) or (right_fork and not left_fork):
                philosophers_with_one_fork += 1
        
        # If all hungry philosophers hold exactly one fork, it's a deadlock
        return philosophers_with_one_fork == len(hungry_philosophers) and philosophers_with_one_fork > 0
    
    def _is_resource_starvation(self, philosophers: List, forks: List) -> bool:
        """Check for resource starvation."""
        hungry_philosophers = [p for p in philosophers if p[1] == 'hungry']
        available_forks = [f for f in forks if f[1] == 'available']
        
        # If there are hungry philosophers but no available forks
        if len(hungry_philosophers) > 0 and len(available_forks) == 0:
            return True
        
        # If there are more hungry philosophers than available forks
        if len(hungry_philosophers) > len(available_forks):
            return True
        
        return False
    
    def _is_circular_wait(self, philosophers: List) -> bool:
        """Check for circular wait condition."""
        # Create a graph of philosophers and their fork dependencies
        G = nx.DiGraph()
        
        for phil in philosophers:
            if phil[1] == 'hungry':  # Only consider hungry philosophers
                phil_id = phil[0]
                left_fork = phil[2]
                right_fork = phil[3]
                
                # Add edges: philosopher -> fork (if holding)
                if left_fork:
                    G.add_edge(phil_id, left_fork)
                if right_fork:
                    G.add_edge(phil_id, right_fork)
                
                # Add edges: fork -> philosopher (if fork is needed)
                if not left_fork or not right_fork:
                    # This philosopher needs forks
                    for other_phil in philosophers:
                        if other_phil[0] != phil_id and other_phil[1] == 'hungry':
                            other_left = other_phil[2]
                            other_right = other_phil[3]
                            
                            # Check if other philosopher holds forks this one needs
                            if (not left_fork and (other_left == left_fork or other_right == left_fork)) or \
                               (not right_fork and (other_left == right_fork or other_right == right_fork)):
                                G.add_edge(phil_id, other_phil[0])
        
        # Check for cycles in the graph
        try:
            cycles = list(nx.simple_cycles(G))
            return len(cycles) > 0
        except:
            return False
    
    def _is_livelock(self, state_id: str) -> bool:
        """Check for livelock (repetitive non-progressing behavior)."""
        # Get outgoing transitions from this state
        self.db.cursor.execute('''
            SELECT to_state, action
            FROM transitions 
            WHERE from_state = ?
        ''', (state_id,))
        transitions = self.db.cursor.fetchall()
        
        # Check if all transitions lead back to similar states
        # This is a simplified livelock detection
        if len(transitions) > 0:
            # Check if all transitions are "go-hungry" actions (no progress)
            all_go_hungry = all(t[1] == 'go-hungry' for t in transitions)
            if all_go_hungry:
                return True
        
        return False
    
    def _classify_deadlock_type(self, state_id: str) -> str:
        """Classify the type of deadlock."""
        self.db.cursor.execute('''
            SELECT node_id, node_state, left_fork_id, right_fork_id
            FROM state_configurations 
            WHERE state_id = ? AND node_type = 'Philosopher'
        ''', (state_id,))
        philosophers = self.db.cursor.fetchall()
        
        self.db.cursor.execute('''
            SELECT node_id, node_state
            FROM state_configurations 
            WHERE state_id = ? AND node_type = 'Fork'
        ''', (state_id,))
        forks = self.db.cursor.fetchall()
        
        if self._is_classic_deadlock(philosophers, forks):
            return "Classic Dining Philosopher Deadlock"
        elif self._is_resource_starvation(philosophers, forks):
            return "Resource Starvation"
        elif self._is_circular_wait(philosophers):
            return "Circular Wait"
        elif self._is_livelock(state_id):
            return "Livelock"
        else:
            return "Unknown Deadlock Type"
    
    def _get_deadlock_reason(self, state_id: str) -> str:
        """Get a detailed reason for the deadlock."""
        self.db.cursor.execute('''
            SELECT node_id, node_state, left_fork_id, right_fork_id
            FROM state_configurations 
            WHERE state_id = ? AND node_type = 'Philosopher'
        ''', (state_id,))
        philosophers = self.db.cursor.fetchall()
        
        hungry_count = sum(1 for p in philosophers if p[1] == 'hungry')
        thinking_count = sum(1 for p in philosophers if p[1] == 'think')
        
        philosophers_with_forks = 0
        for phil in philosophers:
            if phil[1] == 'hungry' and (phil[2] is not None or phil[3] is not None):
                philosophers_with_forks += 1
        
        if hungry_count > 0 and philosophers_with_forks == hungry_count:
            return f"All {hungry_count} hungry philosophers hold exactly one fork each, preventing any from eating"
        elif hungry_count > 0 and philosophers_with_forks == 0:
            return f"{hungry_count} philosophers are hungry but no forks are available"
        elif thinking_count == 0:
            return "No philosophers are thinking, system is stuck"
        else:
            return "Complex deadlock involving multiple resource conflicts"
    
    def _get_affected_philosophers(self, state_id: str) -> List[str]:
        """Get list of philosophers affected by the deadlock."""
        self.db.cursor.execute('''
            SELECT node_id, node_state
            FROM state_configurations 
            WHERE state_id = ? AND node_type = 'Philosopher' AND node_state = 'hungry'
        ''', (state_id,))
        hungry_philosophers = self.db.cursor.fetchall()
        
        return [phil[0] for phil in hungry_philosophers]
    
    def _get_resource_usage(self, state_id: str) -> Dict:
        """Get resource usage statistics for the state."""
        self.db.cursor.execute('''
            SELECT 
                COUNT(CASE WHEN node_type = 'Philosopher' AND node_state = 'hungry' THEN 1 END) as hungry_philosophers,
                COUNT(CASE WHEN node_type = 'Philosopher' AND node_state = 'think' THEN 1 END) as thinking_philosophers,
                COUNT(CASE WHEN node_type = 'Fork' AND node_state = 'available' THEN 1 END) as available_forks,
                COUNT(CASE WHEN node_type = 'Fork' AND node_state != 'available' THEN 1 END) as held_forks
            FROM state_configurations 
            WHERE state_id = ?
        ''', (state_id,))
        
        result = self.db.cursor.fetchone()
        return {
            'hungry_philosophers': result[0],
            'thinking_philosophers': result[1],
            'available_forks': result[2],
            'held_forks': result[3]
        }
    
    def update_database_with_deadlocks(self):
        """Update the database to include deadlock information."""
        # Add deadlock column to states table if it doesn't exist
        try:
            self.db.cursor.execute('''
                ALTER TABLE states ADD COLUMN is_deadlock BOOLEAN DEFAULT 0
            ''')
        except sqlite3.OperationalError:
            # Column already exists
            pass
        
        # Add deadlock type column
        try:
            self.db.cursor.execute('''
                ALTER TABLE states ADD COLUMN deadlock_type TEXT
            ''')
        except sqlite3.OperationalError:
            # Column already exists
            pass
        
        # Add deadlock reason column
        try:
            self.db.cursor.execute('''
                ALTER TABLE states ADD COLUMN deadlock_reason TEXT
            ''')
        except sqlite3.OperationalError:
            # Column already exists
            pass
        
        # Detect deadlocks
        deadlocks = self.detect_deadlocks()
        
        # Update database with deadlock information
        for deadlock in deadlocks:
            self.db.cursor.execute('''
                UPDATE states 
                SET is_deadlock = ?, deadlock_type = ?, deadlock_reason = ?
                WHERE state_id = ?
            ''', (True, deadlock['deadlock_type'], deadlock['deadlock_reason'], deadlock['state_id']))
        
        self.db.conn.commit()
        print(f"Updated database with {len(deadlocks)} deadlock states")
    
    def get_deadlock_summary(self) -> Dict:
        """Get a summary of all deadlock states."""
        self.db.cursor.execute('''
            SELECT state_id, is_start_state, is_final_state, has_reach_property, 
                   deadlock_type, deadlock_reason
            FROM states 
            WHERE is_deadlock = 1
            ORDER BY state_id
        ''')
        deadlock_states = self.db.cursor.fetchall()
        
        # Count by deadlock type
        deadlock_types = {}
        for state in deadlock_states:
            deadlock_type = state[4] or "Unknown"
            if deadlock_type not in deadlock_types:
                deadlock_types[deadlock_type] = 0
            deadlock_types[deadlock_type] += 1
        
        return {
            'total_deadlock_states': len(deadlock_states),
            'deadlock_states': [
                {
                    'state_id': state[0],
                    'is_start_state': state[1],
                    'is_final_state': state[2],
                    'has_reach_property': state[3],
                    'deadlock_type': state[4],
                    'deadlock_reason': state[5]
                }
                for state in deadlock_states
            ],
            'deadlock_type_counts': deadlock_types
        }
    
    def create_deadlock_prevention_rules(self) -> List[str]:
        """Create rules to prevent deadlocks."""
        rules = [
            "1. Resource Ordering: Always acquire forks in a consistent order (e.g., left fork first, then right fork)",
            "2. Timeout Mechanism: Release forks if unable to acquire both within a timeout period",
            "3. Resource Preemption: Allow a philosopher to take a fork from a neighbor if they've been waiting too long",
            "4. Asymmetric Solution: Use different strategies for even and odd numbered philosophers",
            "5. Centralized Coordinator: Use a waiter/coordinator to manage fork allocation",
            "6. Limited Concurrency: Allow only N-1 philosophers to be hungry simultaneously",
            "7. Priority System: Assign priorities to philosophers and allow preemption",
            "8. Random Backoff: Add random delays before retrying fork acquisition"
        ]
        return rules
    
    def analyze_deadlock_patterns(self) -> Dict:
        """Analyze patterns in deadlock states."""
        deadlocks = self.detect_deadlocks()
        
        if not deadlocks:
            return {"message": "No deadlocks detected"}
        
        # Analyze patterns
        patterns = {
            'total_deadlocks': len(deadlocks),
            'deadlock_types': {},
            'resource_usage_patterns': [],
            'philosopher_states': {}
        }
        
        for deadlock in deadlocks:
            # Count deadlock types
            deadlock_type = deadlock['deadlock_type']
            if deadlock_type not in patterns['deadlock_types']:
                patterns['deadlock_types'][deadlock_type] = 0
            patterns['deadlock_types'][deadlock_type] += 1
            
            # Analyze resource usage
            resource_usage = deadlock['resource_usage']
            patterns['resource_usage_patterns'].append(resource_usage)
            
            # Analyze philosopher states
            affected_philosophers = deadlock['affected_philosophers']
            for phil in affected_philosophers:
                if phil not in patterns['philosopher_states']:
                    patterns['philosopher_states'][phil] = 0
                patterns['philosopher_states'][phil] += 1
        
        return patterns
    
    def close(self):
        """Close the database connection."""
        self.db.close()

def main():
    """Main function to demonstrate deadlock detection."""
    print("=== DINING PHILOSOPHER DEADLOCK DETECTION ===\n")
    
    # Create deadlock detector
    detector = DeadlockDetector()
    
    # Update database with deadlock information
    print("1. Updating database with deadlock information...")
    detector.update_database_with_deadlocks()
    
    # Get deadlock summary
    print("\n2. Deadlock Summary:")
    summary = detector.get_deadlock_summary()
    print(f"Total deadlock states: {summary['total_deadlock_states']}")
    
    if summary['total_deadlock_states'] > 0:
        print("\nDeadlock states:")
        for state in summary['deadlock_states']:
            print(f"  State {state['state_id']}: {state['deadlock_type']}")
            print(f"    Reason: {state['deadlock_reason']}")
            if state['is_start_state']:
                print("    -> START STATE")
            if state['is_final_state']:
                print("    -> FINAL STATE")
            if state['has_reach_property']:
                print("    -> Has reach_2 property")
            print()
        
        print("Deadlock type distribution:")
        for deadlock_type, count in summary['deadlock_type_counts'].items():
            print(f"  {deadlock_type}: {count} states")
    
    # Analyze deadlock patterns
    print("\n3. Deadlock Pattern Analysis:")
    patterns = detector.analyze_deadlock_patterns()
    if 'message' not in patterns:
        print(f"Total deadlocks: {patterns['total_deadlocks']}")
        print("Deadlock types:")
        for deadlock_type, count in patterns['deadlock_types'].items():
            print(f"  {deadlock_type}: {count}")
        
        print("\nResource usage patterns in deadlocks:")
        for i, usage in enumerate(patterns['resource_usage_patterns'][:5]):  # Show first 5
            print(f"  Deadlock {i+1}: {usage['hungry_philosophers']} hungry, "
                  f"{usage['available_forks']} available forks, {usage['held_forks']} held forks")
    
    # Generate prevention rules
    print("\n4. Deadlock Prevention Rules:")
    rules = detector.create_deadlock_prevention_rules()
    for rule in rules:
        print(f"  {rule}")
    
    # Create enhanced query for deadlock analysis
    print("\n5. Enhanced Database Queries for Deadlock Analysis:")
    print("""
    -- Find all deadlock states
    SELECT * FROM states WHERE is_deadlock = 1;
    
    -- Find deadlocks by type
    SELECT deadlock_type, COUNT(*) as count 
    FROM states 
    WHERE is_deadlock = 1 
    GROUP BY deadlock_type;
    
    -- Find states with hungry philosophers but no available forks
    SELECT DISTINCT s.state_id, s.deadlock_reason
    FROM states s
    JOIN state_configurations sc ON s.state_id = sc.state_id
    WHERE sc.node_type = 'Philosopher' AND sc.node_state = 'hungry'
    AND NOT EXISTS (
        SELECT 1 FROM state_configurations sc2 
        WHERE sc2.state_id = s.state_id 
        AND sc2.node_type = 'Fork' AND sc2.node_state = 'available'
    );
    
    -- Find transitions that lead to deadlocks
    SELECT t.from_state, t.to_state, t.action, s.deadlock_type
    FROM transitions t
    JOIN states s ON t.to_state = s.state_id
    WHERE s.is_deadlock = 1;
    """)
    
    detector.close()
    print("\nDeadlock detection completed successfully!")

if __name__ == "__main__":
    main() 