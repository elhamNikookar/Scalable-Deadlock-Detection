#!/usr/bin/env python3
"""
Query the Combined Dining Philosopher Database (up_to_40_phil_database.db)
"""

import sqlite3
import pandas as pd
from typing import List, Dict

class CombinedDatabaseQuery:
    def __init__(self, db_path: str = "philosopher_databases/up_to_40_phil_database.db"):
        """Initialize the combined database query interface."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
    
    def get_problem_sizes(self) -> List[int]:
        """Get all available problem sizes."""
        self.cursor.execute('SELECT DISTINCT problem_size FROM problem_sizes ORDER BY problem_size')
        return [row[0] for row in self.cursor.fetchall()]
    
    def get_database_summary(self) -> Dict:
        """Get summary statistics for the combined database."""
        summary = {}
        
        # Get problem sizes
        self.cursor.execute('SELECT COUNT(*) FROM problem_sizes')
        summary['total_problem_sizes'] = self.cursor.fetchone()[0]
        
        # Get total states
        self.cursor.execute('SELECT COUNT(*) FROM states')
        summary['total_states'] = self.cursor.fetchone()[0]
        
        # Get total transitions
        self.cursor.execute('SELECT COUNT(*) FROM transitions')
        summary['total_transitions'] = self.cursor.fetchone()[0]
        
        # Get total configurations
        self.cursor.execute('SELECT COUNT(*) FROM state_configurations')
        summary['total_configurations'] = self.cursor.fetchone()[0]
        
        return summary
    
    def get_states_by_problem_size(self, problem_size: int) -> pd.DataFrame:
        """Get all states for a specific problem size."""
        query = '''
            SELECT state_id, state_name, is_start_state, is_final_state, 
                   has_reach_property, description, problem_size
            FROM states 
            WHERE problem_size = ?
            ORDER BY state_id
        '''
        return pd.read_sql_query(query, self.conn, params=(problem_size,))
    
    def get_transitions_by_problem_size(self, problem_size: int) -> pd.DataFrame:
        """Get all transitions for a specific problem size."""
        query = '''
            SELECT from_state, to_state, action, transition_type, problem_size
            FROM transitions 
            WHERE problem_size = ?
            ORDER BY from_state, to_state
        '''
        return pd.read_sql_query(query, self.conn, params=(problem_size,))
    
    def get_configurations_by_problem_size(self, problem_size: int) -> pd.DataFrame:
        """Get all state configurations for a specific problem size."""
        query = '''
            SELECT state_id, node_id, node_type, node_state, 
                   left_fork_id, right_fork_id, problem_size
            FROM state_configurations 
            WHERE problem_size = ?
            ORDER BY state_id, node_id
        '''
        return pd.read_sql_query(query, self.conn, params=(problem_size,))
    
    def get_deadlock_states_by_problem_size(self, problem_size: int) -> pd.DataFrame:
        """Get deadlock states for a specific problem size."""
        # Check if deadlock column exists
        self.cursor.execute("PRAGMA table_info(states)")
        columns = [col[1] for col in self.cursor.fetchall()]
        has_deadlock = 'is_deadlock' in columns
        
        if has_deadlock:
            query = '''
                SELECT state_id, state_name, is_start_state, is_final_state, 
                       has_reach_property, description, problem_size, is_deadlock, deadlock_type, deadlock_reason
                FROM states 
                WHERE problem_size = ? AND is_deadlock = 1
                ORDER BY state_id
            '''
        else:
            query = '''
                SELECT state_id, state_name, is_start_state, is_final_state, 
                       has_reach_property, description, problem_size
                FROM states 
                WHERE problem_size = ? AND has_reach_property = 1
                ORDER BY state_id
            '''
        
        return pd.read_sql_query(query, self.conn, params=(problem_size,))
    
    def get_problem_size_statistics(self) -> pd.DataFrame:
        """Get statistics for each problem size."""
        # Check if deadlock column exists
        self.cursor.execute("PRAGMA table_info(states)")
        columns = [col[1] for col in self.cursor.fetchall()]
        has_deadlock = 'is_deadlock' in columns
        
        if has_deadlock:
            query = '''
                SELECT 
                    ps.problem_size,
                    ps.num_philosophers,
                    ps.num_forks,
                    COUNT(DISTINCT s.state_id) as num_states,
                    COUNT(DISTINCT t.id) as num_transitions,
                    COUNT(DISTINCT sc.id) as num_configurations,
                    COUNT(CASE WHEN s.is_deadlock = 1 THEN 1 END) as num_deadlocks,
                    ps.created_date
                FROM problem_sizes ps
                LEFT JOIN states s ON ps.problem_size = s.problem_size
                LEFT JOIN transitions t ON ps.problem_size = t.problem_size
                LEFT JOIN state_configurations sc ON ps.problem_size = sc.problem_size
                GROUP BY ps.problem_size
                ORDER BY ps.problem_size
            '''
        else:
            query = '''
                SELECT 
                    ps.problem_size,
                    ps.num_philosophers,
                    ps.num_forks,
                    COUNT(DISTINCT s.state_id) as num_states,
                    COUNT(DISTINCT t.id) as num_transitions,
                    COUNT(DISTINCT sc.id) as num_configurations,
                    0 as num_deadlocks,
                    ps.created_date
                FROM problem_sizes ps
                LEFT JOIN states s ON ps.problem_size = s.problem_size
                LEFT JOIN transitions t ON ps.problem_size = t.problem_size
                LEFT JOIN state_configurations sc ON ps.problem_size = sc.problem_size
                GROUP BY ps.problem_size
                ORDER BY ps.problem_size
            '''
        
        return pd.read_sql_query(query, self.conn)
    
    def get_deadlock_analysis(self) -> pd.DataFrame:
        """Get deadlock analysis across all problem sizes."""
        # Check if deadlock column exists
        self.cursor.execute("PRAGMA table_info(states)")
        columns = [col[1] for col in self.cursor.fetchall()]
        has_deadlock = 'is_deadlock' in columns
        
        if has_deadlock:
            query = '''
                SELECT 
                    problem_size,
                    COUNT(*) as total_states,
                    COUNT(CASE WHEN is_deadlock = 1 THEN 1 END) as deadlock_states,
                    ROUND(COUNT(CASE WHEN is_deadlock = 1 THEN 1 END) * 100.0 / COUNT(*), 2) as deadlock_percentage
                FROM states
                GROUP BY problem_size
                ORDER BY problem_size
            '''
        else:
            query = '''
                SELECT 
                    problem_size,
                    COUNT(*) as total_states,
                    0 as deadlock_states,
                    0.0 as deadlock_percentage
                FROM states
                GROUP BY problem_size
                ORDER BY problem_size
            '''
        
        return pd.read_sql_query(query, self.conn)
    
    def get_transition_patterns(self, problem_size: int) -> pd.DataFrame:
        """Get transition patterns for a specific problem size."""
        query = '''
            SELECT 
                action,
                COUNT(*) as frequency,
                COUNT(DISTINCT from_state) as unique_from_states,
                COUNT(DISTINCT to_state) as unique_to_states
            FROM transitions
            WHERE problem_size = ?
            GROUP BY action
            ORDER BY frequency DESC
        '''
        return pd.read_sql_query(query, self.conn, params=(problem_size,))
    
    def get_state_analysis(self, problem_size: int) -> Dict:
        """Get comprehensive state analysis for a problem size."""
        analysis = {}
        
        # Get basic statistics
        states_df = self.get_states_by_problem_size(problem_size)
        transitions_df = self.get_transitions_by_problem_size(problem_size)
        configs_df = self.get_configurations_by_problem_size(problem_size)
        
        analysis['num_states'] = len(states_df)
        analysis['num_transitions'] = len(transitions_df)
        analysis['num_configurations'] = len(configs_df)
        
        # Get state types
        analysis['start_states'] = len(states_df[states_df['is_start_state'] == 1])
        analysis['final_states'] = len(states_df[states_df['is_final_state'] == 1])
        analysis['reach_property_states'] = len(states_df[states_df['has_reach_property'] == 1])
        
        # Get configuration types
        if len(configs_df) > 0:
            analysis['philosopher_configs'] = len(configs_df[configs_df['node_type'] == 'Philosopher'])
            analysis['fork_configs'] = len(configs_df[configs_df['node_type'] == 'Fork'])
            analysis['thinking_philosophers'] = len(configs_df[
                (configs_df['node_type'] == 'Philosopher') & 
                (configs_df['node_state'] == 'think')
            ])
            analysis['hungry_philosophers'] = len(configs_df[
                (configs_df['node_type'] == 'Philosopher') & 
                (configs_df['node_state'] == 'hungry')
            ])
        
        return analysis
    
    def close(self):
        """Close the database connection."""
        self.conn.close()

def main():
    """Demonstrate querying the combined database."""
    print("=== QUERYING COMBINED DINING PHILOSOPHER DATABASE ===\n")
    
    try:
        query = CombinedDatabaseQuery()
        
        # Get database summary
        print("1. Database Summary:")
        summary = query.get_database_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Get available problem sizes
        print(f"\n2. Available Problem Sizes:")
        problem_sizes = query.get_problem_sizes()
        print(f"  Problem sizes: {problem_sizes}")
        
        # Get statistics for each problem size
        print(f"\n3. Problem Size Statistics:")
        stats_df = query.get_problem_size_statistics()
        print(stats_df.to_string(index=False))
        
        # Get deadlock analysis
        print(f"\n4. Deadlock Analysis:")
        deadlock_df = query.get_deadlock_analysis()
        print(deadlock_df.to_string(index=False))
        
        # Analyze a specific problem size
        sample_size = 5
        print(f"\n5. Analysis for {sample_size} Philosophers:")
        analysis = query.get_state_analysis(sample_size)
        for key, value in analysis.items():
            print(f"  {key}: {value}")
        
        # Get transition patterns for sample size
        print(f"\n6. Transition Patterns for {sample_size} Philosophers:")
        patterns_df = query.get_transition_patterns(sample_size)
        print(patterns_df.to_string(index=False))
        
        # Get states for sample size
        print(f"\n7. States for {sample_size} Philosophers:")
        states_df = query.get_states_by_problem_size(sample_size)
        print(states_df.to_string(index=False))
        
        query.close()
        print("\n✅ Database querying completed successfully!")
        
    except FileNotFoundError:
        print("❌ Combined database not found!")
        print("Please run the database generation script first:")
        print("python generate_philosopher_databases.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 