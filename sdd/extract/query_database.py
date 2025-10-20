import sqlite3
import json
from dining_philosopher_database import DiningPhilosopherDatabase

def query_examples():
    """Demonstrate various queries on the dining philosopher database."""
    
    # Connect to the database
    db = DiningPhilosopherDatabase()
    
    print("=== DINING PHILOSOPHER DATABASE QUERIES ===\n")
    
    # 1. Get all states with their properties
    print("1. ALL STATES:")
    states = db.get_state_summary()
    for state in states:
        print(f"  State {state['state_id']}:")
        print(f"    - Philosophers: {state['philosopher_count']}")
        print(f"    - Forks: {state['fork_count']}")
        if state['is_start_state']:
            print("    - START STATE")
        if state['is_final_state']:
            print("    - FINAL STATE")
        if state['has_reach_property']:
            print("    - Has reach_2 property")
        print()
    
    # 2. Get all transitions
    print("2. ALL TRANSITIONS:")
    transitions = db.get_transition_summary()
    for trans in transitions:
        print(f"  {trans['from_state']} --{trans['action']}--> {trans['to_state']} (freq: {trans['frequency']})")
    print()
    
    # 3. Get detailed information about a specific state
    print("3. DETAILED STATE INFORMATION (s0):")
    state_details = db.get_state_details('s0')
    
    print(f"  State Info: {state_details['state_info']}")
    print("  Configurations:")
    for config in state_details['configurations']:
        print(f"    - {config['node_type']} {config['node_id']}: {config['node_state']}")
        if config['left_fork_id']:
            print(f"      Left fork: {config['left_fork_id']}")
        if config['right_fork_id']:
            print(f"      Right fork: {config['right_fork_id']}")
    
    print("  Incoming transitions:")
    for trans in state_details['incoming_transitions']:
        print(f"    {trans['from_state']} --{trans['action']}--> {trans['to_state']}")
    
    print("  Outgoing transitions:")
    for trans in state_details['outgoing_transitions']:
        print(f"    {trans['from_state']} --{trans['action']}--> {trans['to_state']}")
    print()
    
    # 4. Custom queries
    print("4. CUSTOM QUERIES:")
    
    # Find all philosophers in thinking state
    db.cursor.execute('''
        SELECT state_id, node_id, node_state
        FROM state_configurations 
        WHERE node_type = 'Philosopher' AND node_state = 'think'
    ''')
    thinking_philosophers = db.cursor.fetchall()
    print("  Philosophers in thinking state:")
    for phil in thinking_philosophers:
        print(f"    State {phil[0]}, Philosopher {phil[1]}: {phil[2]}")
    print()
    
    # Find all philosophers in hungry state
    db.cursor.execute('''
        SELECT state_id, node_id, node_state
        FROM state_configurations 
        WHERE node_type = 'Philosopher' AND node_state = 'hungry'
    ''')
    hungry_philosophers = db.cursor.fetchall()
    print("  Philosophers in hungry state:")
    for phil in hungry_philosophers:
        print(f"    State {phil[0]}, Philosopher {phil[1]}: {phil[2]}")
    print()
    
    # Find all fork acquisition actions
    db.cursor.execute('''
        SELECT from_state, to_state, action
        FROM transitions 
        WHERE action IN ('get-left', 'get-right')
        ORDER BY action, from_state
    ''')
    fork_actions = db.cursor.fetchall()
    print("  Fork acquisition actions:")
    for action in fork_actions:
        print(f"    {action[0]} --{action[2]}--> {action[1]}")
    print()
    
    # Find all fork release actions
    db.cursor.execute('''
        SELECT from_state, to_state, action
        FROM transitions 
        WHERE action IN ('release-left', 'release-right')
        ORDER BY action, from_state
    ''')
    release_actions = db.cursor.fetchall()
    print("  Fork release actions:")
    for action in release_actions:
        print(f"    {action[0]} --{action[2]}--> {action[1]}")
    print()
    
    # 5. Find paths to reach state s8 (the final state with reach_2 property)
    print("5. PATHS TO REACH STATE S8:")
    db.cursor.execute('''
        WITH RECURSIVE paths AS (
            SELECT from_state, to_state, action, CAST(from_state || '->' || to_state AS TEXT) as path, 1 as depth
            FROM transitions 
            WHERE to_state = 's8'
            
            UNION ALL
            
            SELECT t.from_state, t.to_state, t.action, 
                   p.path || '->' || t.to_state, p.depth + 1
            FROM transitions t
            JOIN paths p ON t.to_state = p.from_state
            WHERE p.depth < 5  -- Limit path length
        )
        SELECT DISTINCT path, action
        FROM paths
        ORDER BY depth, path
    ''')
    paths = db.cursor.fetchall()
    for path in paths:
        print(f"    Path: {path[0]} (Action: {path[1]})")
    print()
    
    # 6. Find all actions and their categories
    print("6. ALL ACTIONS AND THEIR CATEGORIES:")
    db.cursor.execute('SELECT action_name, description, category FROM actions ORDER BY category, action_name')
    actions = db.cursor.fetchall()
    for action in actions:
        print(f"    {action[0]}: {action[1]} ({action[2]})")
    print()
    
    # 7. Count transitions by action type
    print("7. TRANSITION COUNTS BY ACTION:")
    db.cursor.execute('''
        SELECT action, COUNT(*) as count
        FROM transitions 
        GROUP BY action 
        ORDER BY count DESC
    ''')
    action_counts = db.cursor.fetchall()
    for action_count in action_counts:
        print(f"    {action_count[0]}: {action_count[1]} transitions")
    print()
    
    # 8. Find states with specific properties
    print("8. STATES WITH SPECIAL PROPERTIES:")
    
    # Start states
    db.cursor.execute('SELECT state_id FROM states WHERE is_start_state = 1')
    start_states = [row[0] for row in db.cursor.fetchall()]
    print(f"    Start states: {start_states}")
    
    # Final states
    db.cursor.execute('SELECT state_id FROM states WHERE is_final_state = 1')
    final_states = [row[0] for row in db.cursor.fetchall()]
    print(f"    Final states: {final_states}")
    
    # States with reach_2 property
    db.cursor.execute('SELECT state_id FROM states WHERE has_reach_property = 1')
    reach_states = [row[0] for row in db.cursor.fetchall()]
    print(f"    States with reach_2 property: {reach_states}")
    print()
    
    # Close database
    db.close()

if __name__ == "__main__":
    query_examples() 