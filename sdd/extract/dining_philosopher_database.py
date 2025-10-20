import xml.etree.ElementTree as ET
import sqlite3
import os
import re
from typing import Dict, List, Tuple, Set
import json

class DiningPhilosopherDatabase:
    def __init__(self, db_path: str = "dining_philosopher.db"):
        """Initialize the database and create tables."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.create_tables()
    
    def create_tables(self):
        """Create all necessary tables for the dining philosopher database."""
        
        # States table - stores information about each state
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS states (
                state_id TEXT PRIMARY KEY,
                state_name TEXT,
                is_start_state BOOLEAN,
                is_final_state BOOLEAN,
                has_reach_property BOOLEAN,
                description TEXT,
                problem_size INTEGER DEFAULT NULL
            )
        ''')
        
        # State configurations table - stores the detailed configuration of each state
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS state_configurations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state_id TEXT,
                node_id TEXT,
                node_type TEXT,
                node_state TEXT,
                left_fork_id TEXT,
                right_fork_id TEXT,
                problem_size INTEGER DEFAULT NULL,
                FOREIGN KEY (state_id) REFERENCES states (state_id)
            )
        ''')
        
        # Transitions table - stores state transitions
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_state TEXT,
                to_state TEXT,
                action TEXT,
                transition_type TEXT,
                problem_size INTEGER DEFAULT NULL,
                FOREIGN KEY (from_state) REFERENCES states (state_id),
                FOREIGN KEY (to_state) REFERENCES states (state_id)
            )
        ''')
        
        # Actions table - stores all possible actions
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS actions (
                action_name TEXT PRIMARY KEY,
                description TEXT,
                category TEXT
            )
        ''')
        
        # Nodes table - stores information about individual nodes
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state_id TEXT,
                node_id TEXT,
                node_type TEXT,
                attributes TEXT,
                FOREIGN KEY (state_id) REFERENCES states (state_id)
            )
        ''')
        
        # Edges table - stores information about edges between nodes
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state_id TEXT,
                from_node TEXT,
                to_node TEXT,
                edge_label TEXT,
                FOREIGN KEY (state_id) REFERENCES states (state_id)
            )
        ''')
        
        self.conn.commit()
    
    def parse_gst_file(self, file_path: str) -> Dict:
        """Parse a GST file and extract state information."""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Extract state ID from filename
        state_id = os.path.splitext(os.path.basename(file_path))[0]
        
        # Find the graph element
        graph = root.find('.//{http://www.gupro.de/GXL/gxl-1.0.dtd}graph')
        if graph is None:
            return {}
        
        # Extract nodes and edges
        nodes = {}
        edges = []
        
        # Parse nodes
        for node in graph.findall('.//{http://www.gupro.de/GXL/gxl-1.0.dtd}node'):
            node_id = node.get('id')
            nodes[node_id] = {'id': node_id, 'attributes': {}}
        
        # Parse edges
        for edge in graph.findall('.//{http://www.gupro.de/GXL/gxl-1.0.dtd}edge'):
            from_node = edge.get('from')
            to_node = edge.get('to')
            
            # Get edge label
            label_attr = edge.find('.//{http://www.gupro.de/GXL/gxl-1.0.dtd}attr[@name="label"]')
            if label_attr is not None:
                label_elem = label_attr.find('.//{http://www.gupro.de/GXL/gxl-1.0.dtd}string')
                if label_elem is not None:
                    edge_label = label_elem.text
                else:
                    edge_label = ""
            else:
                edge_label = ""
            
            edges.append({
                'from': from_node,
                'to': to_node,
                'label': edge_label
            })
        
        return {
            'state_id': state_id,
            'nodes': nodes,
            'edges': edges
        }
    
    def parse_dot_file(self, file_path: str) -> Dict:
        """Parse a DOT file and extract transition information."""
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract state definitions
        state_pattern = r'(\w+)\[label=<\\N<br/>(?:<i>(\w+)</i><br/>)?>\]'
        states = re.findall(state_pattern, content)
        
        # Extract transitions
        transition_pattern = r'(\w+)->(\w+)\[label=<(\w+)>\]'
        transitions = re.findall(transition_pattern, content)
        
        return {
            'states': states,
            'transitions': transitions
        }
    
    def parse_gxl_file(self, file_path: str) -> Dict:
        """Parse a GXL file and extract labeled transition system information."""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Find the graph element
        graph = root.find('.//{http://www.gupro.de/GXL/gxl-1.0.dtd}graph')
        if graph is None:
            return {}
        
        nodes = {}
        edges = []
        
        # Parse nodes
        for node in graph.findall('.//{http://www.gupro.de/GXL/gxl-1.0.dtd}node'):
            node_id = node.get('id')
            nodes[node_id] = {'id': node_id, 'attributes': {}}
        
        # Parse edges
        for edge in graph.findall('.//{http://www.gupro.de/GXL/gxl-1.0.dtd}edge'):
            from_node = edge.get('from')
            to_node = edge.get('to')
            
            # Get edge label
            label_attr = edge.find('.//{http://www.gupro.de/GXL/gxl-1.0.dtd}attr[@name="label"]')
            if label_attr is not None:
                label_elem = label_attr.find('.//{http://www.gupro.de/GXL/gxl-1.0.dtd}string')
                if label_elem is not None:
                    edge_label = label_elem.text
                else:
                    edge_label = ""
            else:
                edge_label = ""
            
            edges.append({
                'from': from_node,
                'to': to_node,
                'label': edge_label
            })
        
        return {
            'nodes': nodes,
            'edges': edges
        }
    
    def analyze_state_configuration(self, gst_data: Dict) -> Dict:
        """Analyze the configuration of philosophers and forks in a state."""
        state_id = gst_data['state_id']
        nodes = gst_data['nodes']
        edges = gst_data['edges']
        
        # Create a mapping of node relationships
        node_relationships = {}
        node_attributes = {}
        
        # Process edges to understand relationships
        for edge in edges:
            from_node = edge['from']
            to_node = edge['to']
            label = edge['label']
            
            if from_node == to_node:
                # Self-loop - this is an attribute of the node
                if from_node not in node_attributes:
                    node_attributes[from_node] = []
                node_attributes[from_node].append(label)
            else:
                # Relationship between nodes
                if from_node not in node_relationships:
                    node_relationships[from_node] = {}
                node_relationships[from_node][label] = to_node
        
        # Analyze philosophers and forks
        philosophers = []
        forks = []
        
        for node_id, attributes in node_attributes.items():
            if 'Phil' in attributes:
                # This is a philosopher
                phil_info = {
                    'node_id': node_id,
                    'state': next((attr for attr in attributes if attr in ['think', 'hungry']), 'unknown'),
                    'left_fork': node_relationships.get(node_id, {}).get('left'),
                    'right_fork': node_relationships.get(node_id, {}).get('right')
                }
                philosophers.append(phil_info)
            elif 'Fork' in attributes:
                # This is a fork
                fork_info = {
                    'node_id': node_id,
                    'status': 'available'  # Default assumption
                }
                forks.append(fork_info)
        
        return {
            'state_id': state_id,
            'philosophers': philosophers,
            'forks': forks,
            'node_attributes': node_attributes,
            'node_relationships': node_relationships
        }
    
    def insert_state_data(self, state_analysis: Dict):
        """Insert state data into the database."""
        state_id = state_analysis['state_id']
        
        # Insert into states table
        self.cursor.execute('''
            INSERT OR REPLACE INTO states (state_id, state_name, is_start_state, is_final_state, has_reach_property)
            VALUES (?, ?, ?, ?, ?)
        ''', (state_id, state_id, False, False, False))
        
        # Insert philosopher configurations
        for phil in state_analysis['philosophers']:
            self.cursor.execute('''
                INSERT INTO state_configurations 
                (state_id, node_id, node_type, node_state, left_fork_id, right_fork_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                state_id,
                phil['node_id'],
                'Philosopher',
                phil['state'],
                phil['left_fork'],
                phil['right_fork']
            ))
        
        # Insert fork configurations
        for fork in state_analysis['forks']:
            self.cursor.execute('''
                INSERT INTO state_configurations 
                (state_id, node_id, node_type, node_state, left_fork_id, right_fork_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                state_id,
                fork['node_id'],
                'Fork',
                fork['status'],
                None,
                None
            ))
    
    def insert_transition_data(self, dot_data: Dict):
        """Insert transition data from DOT file."""
        for from_state, to_state, action in dot_data['transitions']:
            self.cursor.execute('''
                INSERT OR REPLACE INTO transitions (from_state, to_state, action, transition_type)
                VALUES (?, ?, ?, ?)
            ''', (from_state, to_state, action, 'state_transition'))
    
    def insert_lts_data(self, gxl_data: Dict):
        """Insert labeled transition system data."""
        for edge in gxl_data['edges']:
            from_node = edge['from']
            to_node = edge['to']
            label = edge['label']
            
            # Check if this is a state transition
            if label.startswith('s') and label[1:].isdigit():
                # This is a state identifier
                continue
            
            # Check if this is a special property
            if label in ['start', 'final', 'reach_2']:
                # Update state properties
                if label == 'start':
                    self.cursor.execute('''
                        UPDATE states SET is_start_state = ? WHERE state_id = ?
                    ''', (True, from_node))
                elif label == 'final':
                    self.cursor.execute('''
                        UPDATE states SET is_final_state = ? WHERE state_id = ?
                    ''', (True, from_node))
                elif label == 'reach_2':
                    self.cursor.execute('''
                        UPDATE states SET has_reach_property = ? WHERE state_id = ?
                    ''', (True, from_node))
            else:
                # This is an action transition
                self.cursor.execute('''
                    INSERT OR REPLACE INTO transitions (from_state, to_state, action, transition_type)
                    VALUES (?, ?, ?, ?)
                ''', (from_node, to_node, label, 'lts_transition'))
    
    def insert_action_data(self):
        """Insert predefined actions into the actions table."""
        actions = [
            ('go-hungry', 'Philosopher becomes hungry', 'hunger'),
            ('get-left', 'Philosopher picks up left fork', 'fork_acquisition'),
            ('get-right', 'Philosopher picks up right fork', 'fork_acquisition'),
            ('release-left', 'Philosopher releases left fork', 'fork_release'),
            ('release-right', 'Philosopher releases right fork', 'fork_release'),
            ('think', 'Philosopher is thinking', 'philosophy'),
            ('hungry', 'Philosopher is hungry', 'hunger'),
            ('start', 'Initial state', 'system'),
            ('final', 'Final state', 'system'),
            ('reach_2', 'Reachability property', 'property')
        ]
        
        for action_name, description, category in actions:
            self.cursor.execute('''
                INSERT OR REPLACE INTO actions (action_name, description, category)
                VALUES (?, ?, ?)
            ''', (action_name, description, category))
    
    def process_all_files(self, directory: str = "."):
        """Process all relevant files in the directory."""
        
        # Process GST files (state configurations)
        gst_files = [f for f in os.listdir(directory) if f.endswith('.gst')]
        for gst_file in sorted(gst_files):
            print(f"Processing GST file: {gst_file}")
            gst_data = self.parse_gst_file(os.path.join(directory, gst_file))
            if gst_data:
                state_analysis = self.analyze_state_configuration(gst_data)
                self.insert_state_data(state_analysis)
        
        # Process DOT file (state transitions)
        dot_files = [f for f in os.listdir(directory) if f.endswith('.dot')]
        for dot_file in dot_files:
            print(f"Processing DOT file: {dot_file}")
            dot_data = self.parse_dot_file(os.path.join(directory, dot_file))
            self.insert_transition_data(dot_data)
        
        # Process GXL file (labeled transition system)
        gxl_files = [f for f in os.listdir(directory) if f.endswith('.gxl')]
        for gxl_file in gxl_files:
            print(f"Processing GXL file: {gxl_file}")
            gxl_data = self.parse_gxl_file(os.path.join(directory, gxl_file))
            self.insert_lts_data(gxl_data)
        
        # Insert action data
        self.insert_action_data()
        
        self.conn.commit()
        print("Database creation completed!")
    
    def get_state_summary(self) -> List[Dict]:
        """Get a summary of all states."""
        self.cursor.execute('''
            SELECT s.state_id, s.is_start_state, s.is_final_state, s.has_reach_property,
                   COUNT(CASE WHEN sc.node_type = 'Philosopher' THEN 1 END) as philosopher_count,
                   COUNT(CASE WHEN sc.node_type = 'Fork' THEN 1 END) as fork_count
            FROM states s
            LEFT JOIN state_configurations sc ON s.state_id = sc.state_id
            GROUP BY s.state_id
            ORDER BY s.state_id
        ''')
        
        return [dict(zip([col[0] for col in self.cursor.description], row)) 
                for row in self.cursor.fetchall()]
    
    def get_transition_summary(self) -> List[Dict]:
        """Get a summary of all transitions."""
        self.cursor.execute('''
            SELECT from_state, to_state, action, COUNT(*) as frequency
            FROM transitions
            GROUP BY from_state, to_state, action
            ORDER BY from_state, to_state
        ''')
        
        return [dict(zip([col[0] for col in self.cursor.description], row)) 
                for row in self.cursor.fetchall()]
    
    def get_state_details(self, state_id: str) -> Dict:
        """Get detailed information about a specific state."""
        # Get state info
        self.cursor.execute('''
            SELECT * FROM states WHERE state_id = ?
        ''', (state_id,))
        state_info = dict(zip([col[0] for col in self.cursor.description], 
                             self.cursor.fetchone() or []))
        
        # Get state configuration
        self.cursor.execute('''
            SELECT * FROM state_configurations WHERE state_id = ?
        ''', (state_id,))
        configurations = [dict(zip([col[0] for col in self.cursor.description], row)) 
                        for row in self.cursor.fetchall()]
        
        # Get incoming transitions
        self.cursor.execute('''
            SELECT * FROM transitions WHERE to_state = ?
        ''', (state_id,))
        incoming = [dict(zip([col[0] for col in self.cursor.description], row)) 
                   for row in self.cursor.fetchall()]
        
        # Get outgoing transitions
        self.cursor.execute('''
            SELECT * FROM transitions WHERE from_state = ?
        ''', (state_id,))
        outgoing = [dict(zip([col[0] for col in self.cursor.description], row)) 
                   for row in self.cursor.fetchall()]
        
        return {
            'state_info': state_info,
            'configurations': configurations,
            'incoming_transitions': incoming,
            'outgoing_transitions': outgoing
        }
    
    def export_to_json(self, output_file: str = "dining_philosopher_data.json"):
        """Export all database data to JSON format."""
        data = {
            'states': self.get_state_summary(),
            'transitions': self.get_transition_summary(),
            'actions': []
        }
        
        # Get actions
        self.cursor.execute('SELECT * FROM actions')
        data['actions'] = [dict(zip([col[0] for col in self.cursor.description], row)) 
                          for row in self.cursor.fetchall()]
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Data exported to {output_file}")
    
    def close(self):
        """Close the database connection."""
        self.conn.close()

def main():
    """Main function to create the database and process all files."""
    print("Creating Dining Philosopher Database...")
    
    # Create database instance
    db = DiningPhilosopherDatabase()
    
    # Process all files
    db.process_all_files()
    
    # Print summary
    print("\n=== DATABASE SUMMARY ===")
    
    # State summary
    states = db.get_state_summary()
    print(f"\nTotal States: {len(states)}")
    for state in states:
        print(f"State {state['state_id']}: {state['philosopher_count']} philosophers, "
              f"{state['fork_count']} forks")
        if state['is_start_state']:
            print("  -> START STATE")
        if state['is_final_state']:
            print("  -> FINAL STATE")
        if state['has_reach_property']:
            print("  -> Has reach_2 property")
    
    # Transition summary
    transitions = db.get_transition_summary()
    print(f"\nTotal Transitions: {len(transitions)}")
    for trans in transitions[:10]:  # Show first 10
        print(f"{trans['from_state']} --{trans['action']}--> {trans['to_state']} "
              f"(frequency: {trans['frequency']})")
    
    # Export to JSON
    db.export_to_json()
    
    # Close database
    db.close()
    
    print("\nDatabase creation completed successfully!")
    print("Files created:")
    print("- dining_philosopher.db (SQLite database)")
    print("- dining_philosopher_data.json (JSON export)")

if __name__ == "__main__":
    main() 