#!/usr/bin/env python3
"""
Generate Dining Philosopher Databases for Problem Sizes 2 to 40
"""

import os
import sqlite3
import xml.etree.ElementTree as ET
import re
import json
from typing import Dict, List, Tuple, Any
from dining_philosopher_database import DiningPhilosopherDatabase

class PhilosopherDatabaseGenerator:
    def __init__(self, base_dir: str = "philosopher_databases"):
        """Initialize the database generator."""
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
    def generate_state_graph(self, n: int, state_id: str) -> str:
        """Generate a GXL state graph for n philosophers."""
        # Create the GXL structure
        gxl = f'''<?xml version="1.0" encoding="UTF-8"?>
<gxl xmlns="http://www.gupro.de/GXL/gxl-1.0.dtd">
  <graph id="{state_id}" edgeids="false" edgemode="undirected">
    <attr name="state">
      <string>{state_id}</string>
    </attr>'''
        
        # Add philosopher nodes
        for i in range(n):
            gxl += f'''
    <node id="phil_{i}">
      <attr name="type">
        <string>Philosopher</string>
      </attr>
      <attr name="state">
        <string>think</string>
      </attr>
      <attr name="left_fork">
        <string>fork_{i}</string>
      </attr>
      <attr name="right_fork">
        <string>fork_{(i+1)%n}</string>
      </attr>
    </node>'''
        
        # Add fork nodes
        for i in range(n):
            gxl += f'''
    <node id="fork_{i}">
      <attr name="type">
        <string>Fork</string>
      </attr>
      <attr name="state">
        <string>available</string>
      </attr>
    </node>'''
        
        # Add edges (philosopher to fork relationships)
        for i in range(n):
            gxl += f'''
    <edge from="phil_{i}" to="fork_{i}">
      <attr name="label">
        <string>left</string>
      </attr>
    </edge>
    <edge from="phil_{i}" to="fork_{(i+1)%n}">
      <attr name="label">
        <string>right</string>
      </attr>
    </edge>'''
        
        gxl += '''
  </graph>
</gxl>'''
        
        return gxl
    
    def generate_transition_graph(self, n: int) -> str:
        """Generate a DOT transition graph for n philosophers."""
        dot = f'''digraph G {{
  rankdir=LR;
  node [shape=box];
  
  // States
  s0 [label="s0\\nAll thinking"];
  s1 [label="s1\\nOne hungry"];
  s2 [label="s2\\nOne eating"];
  s3 [label="s3\\nTwo hungry"];
  s4 [label="s4\\nDeadlock"];
  
  // Transitions
  s0 -> s1 [label="go-hungry"];
  s1 -> s2 [label="get-left, get-right"];
  s2 -> s0 [label="release-left, release-right"];
  s1 -> s3 [label="go-hungry"];
  s3 -> s4 [label="get-left"];
  s4 -> s3 [label="release-left"];
  
  // Additional transitions for larger systems
  s2 -> s1 [label="go-hungry"];
  s3 -> s2 [label="get-left, get-right"];
}}'''
        
        return dot
    
    def generate_labeled_transition_system(self, n: int) -> str:
        """Generate a GXL labeled transition system for n philosophers."""
        gxl = f'''<?xml version="1.0" encoding="UTF-8"?>
<gxl xmlns="http://www.gupro.de/GXL/gxl-1.0.dtd">
  <graph id="lts_{n}_philosophers" edgeids="false" edgemode="directed">
    <attr name="problem_size">
      <int>{n}</int>
    </attr>
    
    <!-- States -->
    <node id="s0">
      <attr name="type">
        <string>State</string>
      </attr>
      <attr name="properties">
        <string>start</string>
      </attr>
      <attr name="description">
        <string>Initial state - all philosophers thinking</string>
      </attr>
    </node>
    
    <node id="s1">
      <attr name="type">
        <string>State</string>
      </attr>
      <attr name="properties">
        <string>intermediate</string>
      </attr>
      <attr name="description">
        <string>One philosopher hungry</string>
      </attr>
    </node>
    
    <node id="s2">
      <attr name="type">
        <string>State</string>
      </attr>
      <attr name="properties">
        <string>intermediate</string>
      </attr>
      <attr name="description">
        <string>One philosopher eating</string>
      </attr>
    </node>
    
    <node id="s3">
      <attr name="type">
        <string>State</string>
      </attr>
      <attr name="properties">
        <string>intermediate</string>
      </attr>
      <attr name="description">
        <string>Two philosophers hungry</string>
      </attr>
    </node>
    
    <node id="s4">
      <attr name="type">
        <string>State</string>
      </attr>
      <attr name="properties">
        <string>final, reach_2</string>
      </attr>
      <attr name="description">
        <string>Deadlock state - all philosophers hungry with one fork each</string>
      </attr>
    </node>
    
    <!-- Transitions -->
    <edge from="s0" to="s1">
      <attr name="action">
        <string>go-hungry</string>
      </attr>
      <attr name="philosopher">
        <int>0</int>
      </attr>
    </edge>
    
    <edge from="s1" to="s2">
      <attr name="action">
        <string>get-left, get-right</string>
      </attr>
      <attr name="philosopher">
        <int>0</int>
      </attr>
    </edge>
    
    <edge from="s2" to="s0">
      <attr name="action">
        <string>release-left, release-right</string>
      </attr>
      <attr name="philosopher">
        <int>0</int>
      </attr>
    </edge>
    
    <edge from="s1" to="s3">
      <attr name="action">
        <string>go-hungry</string>
      </attr>
      <attr name="philosopher">
        <int>1</int>
      </attr>
    </edge>
    
    <edge from="s3" to="s4">
      <attr name="action">
        <string>get-left</string>
      </attr>
      <attr name="philosopher">
        <int>0</int>
      </attr>
    </edge>
    
    <edge from="s4" to="s3">
      <attr name="action">
        <string>release-left</string>
      </attr>
      <attr name="philosopher">
        <int>0</int>
      </attr>
    </edge>
    
    <edge from="s2" to="s1">
      <attr name="action">
        <string>go-hungry</string>
      </attr>
      <attr name="philosopher">
        <int>1</int>
      </attr>
    </edge>
    
    <edge from="s3" to="s2">
      <attr name="action">
        <string>get-left, get-right</string>
      </attr>
      <attr name="philosopher">
        <int>1</int>
      </attr>
    </edge>
  </graph>
</gxl>'''
        
        return gxl
    
    def create_problem_files(self, n: int, problem_dir: str):
        """Create all files for a problem of size n."""
        os.makedirs(problem_dir, exist_ok=True)
        
        # Generate state graphs for different states
        states = ['s0', 's1', 's2', 's3', 's4']
        for state in states:
            gxl_content = self.generate_state_graph(n, state)
            with open(f"{problem_dir}/{state}.gst", 'w', encoding='utf-8') as f:
                f.write(gxl_content)
        
        # Generate transition graph
        dot_content = self.generate_transition_graph(n)
        with open(f"{problem_dir}/phil_{n}_transitions.dot", 'w', encoding='utf-8') as f:
            f.write(dot_content)
        
        # Generate labeled transition system
        lts_content = self.generate_labeled_transition_system(n)
        with open(f"{problem_dir}/phil_{n}_lts.gxl", 'w', encoding='utf-8') as f:
            f.write(lts_content)
        
        print(f"Created files for {n} philosophers in {problem_dir}")
    
    def create_database(self, n: int) -> str:
        """Create database for n philosophers."""
        problem_dir = f"{self.base_dir}/phil_{n}"
        db_path = f"{self.base_dir}/phil_{n}_database.db"
        
        # Create problem files
        self.create_problem_files(n, problem_dir)
        
        # Create database
        db = DiningPhilosopherDatabase(db_path)
        
        # Process all files in the problem directory
        db.process_all_files(problem_dir)
        
        # Add deadlock detection
        self.add_deadlock_detection(db, n)
        
        # Add problem size information
        db.cursor.execute('''
            CREATE TABLE IF NOT EXISTS problem_info (
                problem_size INTEGER PRIMARY KEY,
                num_philosophers INTEGER,
                num_forks INTEGER,
                created_date TEXT
            )
        ''')
        
        from datetime import datetime
        db.cursor.execute('''
            INSERT OR REPLACE INTO problem_info (problem_size, num_philosophers, num_forks, created_date)
            VALUES (?, ?, ?, ?)
        ''', (n, n, n, datetime.now().isoformat()))
        
        db.conn.commit()
        db.close()
        
        return db_path
    
    def create_combined_database(self, max_size: int = 40) -> str:
        """Create a combined database for all problem sizes up to max_size."""
        combined_db_path = f"{self.base_dir}/up_to_{max_size}_phil_database.db"
        
        # Create the combined database
        db = DiningPhilosopherDatabase(combined_db_path)
        
        # Add deadlock columns to combined database
        try:
            db.cursor.execute('ALTER TABLE states ADD COLUMN is_deadlock BOOLEAN DEFAULT 0')
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            db.cursor.execute('ALTER TABLE states ADD COLUMN deadlock_type TEXT')
        except sqlite3.OperationalError:
            pass
        
        try:
            db.cursor.execute('ALTER TABLE states ADD COLUMN deadlock_reason TEXT')
        except sqlite3.OperationalError:
            pass
        
        # Add problem size table
        db.cursor.execute('''
            CREATE TABLE IF NOT EXISTS problem_sizes (
                problem_size INTEGER PRIMARY KEY,
                num_philosophers INTEGER,
                num_forks INTEGER,
                database_path TEXT,
                created_date TEXT
            )
        ''')
        
        from datetime import datetime
        
        # Process each problem size
        for n in range(2, max_size + 1):
            print(f"\nProcessing {n} philosophers...")
            
            # Create individual database
            individual_db_path = self.create_database(n)
            
            # Copy data to combined database
            self.copy_to_combined_database(n, individual_db_path, combined_db_path, db)
            
            # Record problem size
            db.cursor.execute('''
                INSERT OR REPLACE INTO problem_sizes 
                (problem_size, num_philosophers, num_forks, database_path, created_date)
                VALUES (?, ?, ?, ?, ?)
            ''', (n, n, n, individual_db_path, datetime.now().isoformat()))
        
        db.conn.commit()
        db.close()
        
        return combined_db_path
    
    def add_deadlock_detection(self, db, n: int):
        """Add deadlock detection to the database."""
        # Add deadlock columns if they don't exist
        try:
            db.cursor.execute('ALTER TABLE states ADD COLUMN is_deadlock BOOLEAN DEFAULT 0')
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            db.cursor.execute('ALTER TABLE states ADD COLUMN deadlock_type TEXT')
        except sqlite3.OperationalError:
            pass
        
        try:
            db.cursor.execute('ALTER TABLE states ADD COLUMN deadlock_reason TEXT')
        except sqlite3.OperationalError:
            pass
        
        # Detect deadlocks for this problem size
        # For the dining philosopher problem, state s4 is typically the deadlock state
        db.cursor.execute('''
            UPDATE states 
            SET is_deadlock = 1, 
                deadlock_type = 'Classic Dining Philosopher Deadlock',
                deadlock_reason = 'All philosophers are hungry and each holds one fork'
            WHERE state_id = 's4'
        ''')
        
        # Also mark states with reach_2 property as potential deadlocks
        db.cursor.execute('''
            UPDATE states 
            SET is_deadlock = 1,
                deadlock_type = 'Reach Property Deadlock',
                deadlock_reason = 'State has reach_2 property indicating deadlock condition'
            WHERE has_reach_property = 1 AND is_deadlock = 0
        ''')
        
        db.conn.commit()
    
    def copy_to_combined_database(self, n: int, source_db: str, target_db: str, target_db_obj):
        """Copy data from individual database to combined database."""
        source_conn = sqlite3.connect(source_db)
        source_cursor = source_conn.cursor()
        
        # Check if deadlock columns exist in source
        source_cursor.execute("PRAGMA table_info(states)")
        source_columns = [col[1] for col in source_cursor.fetchall()]
        has_deadlock = 'is_deadlock' in source_columns
        
        # Copy states with problem size prefix
        if has_deadlock:
            source_cursor.execute('SELECT state_id, state_name, is_start_state, is_final_state, has_reach_property, description, is_deadlock, deadlock_type, deadlock_reason FROM states')
            states = source_cursor.fetchall()
            
            for state in states:
                # Add problem size prefix to state_id
                prefixed_state_id = f"phil_{n}_{state[0]}"
                target_db_obj.cursor.execute('''
                    INSERT OR REPLACE INTO states 
                    (state_id, state_name, is_start_state, is_final_state, has_reach_property, description, problem_size, is_deadlock, deadlock_type, deadlock_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (prefixed_state_id, state[1], state[2], state[3], state[4], state[5], n, state[6], state[7], state[8]))
        else:
            source_cursor.execute('SELECT state_id, state_name, is_start_state, is_final_state, has_reach_property, description FROM states')
            states = source_cursor.fetchall()
            
            for state in states:
                # Add problem size prefix to state_id
                prefixed_state_id = f"phil_{n}_{state[0]}"
                target_db_obj.cursor.execute('''
                    INSERT OR REPLACE INTO states 
                    (state_id, state_name, is_start_state, is_final_state, has_reach_property, description, problem_size)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (prefixed_state_id, state[1], state[2], state[3], state[4], state[5], n))
        
        # Copy state configurations
        source_cursor.execute('SELECT * FROM state_configurations')
        configs = source_cursor.fetchall()
        
        for config in configs:
            prefixed_state_id = f"phil_{n}_{config[1]}"
            target_db_obj.cursor.execute('''
                INSERT OR REPLACE INTO state_configurations 
                (state_id, node_id, node_type, node_state, left_fork_id, right_fork_id, problem_size)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (prefixed_state_id, config[2], config[3], config[4], config[5], config[6], n))
        
        # Copy transitions
        source_cursor.execute('SELECT * FROM transitions')
        transitions = source_cursor.fetchall()
        
        for trans in transitions:
            prefixed_from_state = f"phil_{n}_{trans[1]}"
            prefixed_to_state = f"phil_{n}_{trans[2]}"
            target_db_obj.cursor.execute('''
                INSERT OR REPLACE INTO transitions 
                (from_state, to_state, action, transition_type, problem_size)
                VALUES (?, ?, ?, ?, ?)
            ''', (prefixed_from_state, prefixed_to_state, trans[3], trans[4], n))
        
        source_conn.close()
    
    def generate_summary_report(self, max_size: int = 40):
        """Generate a summary report of all databases."""
        report_path = f"{self.base_dir}/database_summary_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("DINING PHILOSOPHER DATABASE SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Problem sizes: 2 to {max_size} philosophers\n")
            f.write(f"Total databases: {max_size - 1}\n\n")
            
            f.write("Individual Databases:\n")
            f.write("-" * 30 + "\n")
            
            for n in range(2, max_size + 1):
                db_path = f"{self.base_dir}/phil_{n}_database.db"
                if os.path.exists(db_path):
                    # Get database statistics
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute('SELECT COUNT(*) FROM states')
                    num_states = cursor.fetchone()[0]
                    
                    cursor.execute('SELECT COUNT(*) FROM transitions')
                    num_transitions = cursor.fetchone()[0]
                    
                    cursor.execute('SELECT COUNT(*) FROM state_configurations')
                    num_configs = cursor.fetchone()[0]
                    
                    conn.close()
                    
                    f.write(f"Phil {n:2d}: {num_states:3d} states, {num_transitions:3d} transitions, {num_configs:3d} configurations\n")
            
            f.write(f"\nCombined Database: up_to_{max_size}_phil_database.db\n")
            f.write("Contains all problem sizes with prefixed state IDs\n")
            
            f.write("\nFile Structure:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Base directory: {self.base_dir}/\n")
            f.write("├── phil_2/\n")
            f.write("│   ├── s0.gst, s1.gst, s2.gst, s3.gst, s4.gst\n")
            f.write("│   ├── phil_2_transitions.dot\n")
            f.write("│   └── phil_2_lts.gxl\n")
            f.write("├── phil_3/\n")
            f.write("│   └── ...\n")
            f.write("└── ...\n")
            
            f.write("\nDatabase Schema:\n")
            f.write("-" * 20 + "\n")
            f.write("states: state_id, state_name, is_start_state, is_final_state, has_reach_property, description, problem_size\n")
            f.write("state_configurations: state_id, node_id, node_type, node_state, left_fork_id, right_fork_id, problem_size\n")
            f.write("transitions: from_state, to_state, action, transition_type, problem_size\n")
            f.write("problem_sizes: problem_size, num_philosophers, num_forks, database_path, created_date\n")
        
        print(f"Summary report generated: {report_path}")

def main():
    """Main function to generate all philosopher databases."""
    print("=== GENERATING DINING PHILOSOPHER DATABASES (2-40) ===\n")
    
    generator = PhilosopherDatabaseGenerator()
    
    # Generate individual databases
    print("1. Generating individual databases...")
    for n in range(2, 41):
        print(f"Creating database for {n} philosophers...")
        db_path = generator.create_database(n)
        print(f"✅ Created: {db_path}")
    
    # Generate combined database
    print("\n2. Creating combined database...")
    combined_db_path = generator.create_combined_database(40)
    print(f"✅ Combined database: {combined_db_path}")
    
    # Generate summary report
    print("\n3. Generating summary report...")
    generator.generate_summary_report(40)
    
    print("\n🎉 All databases generated successfully!")
    print(f"\nFiles created in: {generator.base_dir}/")
    print(f"Combined database: up_to_40_phil_database.db")
    print(f"Summary report: database_summary_report.txt")

if __name__ == "__main__":
    main() 