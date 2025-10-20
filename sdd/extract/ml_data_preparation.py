import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import json
from dining_philosopher_database import DiningPhilosopherDatabase
import networkx as nx
from typing import Dict, List, Tuple, Any
import pickle

class MLDataPreparation:
    def __init__(self, db_path: str = "dining_philosopher.db"):
        """Initialize the ML data preparation class."""
        self.db = DiningPhilosopherDatabase(db_path)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def create_state_features(self) -> pd.DataFrame:
        """Create feature matrix for state classification."""
        # Get all state configurations
        query = '''
            SELECT 
                s.state_id,
                s.is_start_state,
                s.is_final_state,
                s.has_reach_property,
                s.is_deadlock,
                s.deadlock_type,
                s.deadlock_reason,
                sc.node_id,
                sc.node_type,
                sc.node_state,
                sc.left_fork_id,
                sc.right_fork_id
            FROM states s
            LEFT JOIN state_configurations sc ON s.state_id = sc.state_id
            ORDER BY s.state_id, sc.node_id
        '''
        
        self.db.cursor.execute(query)
        results = self.db.cursor.fetchall()
        
        # Create features for each state
        state_features = {}
        
        for row in results:
            state_id = row[0]
            if state_id not in state_features:
                state_features[state_id] = {
                    'state_id': state_id,
                    'is_start_state': row[1],
                    'is_final_state': row[2],
                    'has_reach_property': row[3],
                    'is_deadlock': row[4],
                    'deadlock_type': row[5],
                    'deadlock_reason': row[6],
                    'philosopher_count': 0,
                    'fork_count': 0,
                    'thinking_philosophers': 0,
                    'hungry_philosophers': 0,
                    'available_forks': 0,
                    'held_forks': 0,
                    'philosopher_states': [],
                    'fork_assignments': []
                }
            
            node_type = row[5]
            node_state = row[6]
            
            if node_type == 'Philosopher':
                state_features[state_id]['philosopher_count'] += 1
                if node_state == 'think':
                    state_features[state_id]['thinking_philosophers'] += 1
                elif node_state == 'hungry':
                    state_features[state_id]['hungry_philosophers'] += 1
                state_features[state_id]['philosopher_states'].append(node_state)
                
                # Count fork assignments
                if row[7]:  # left_fork_id
                    state_features[state_id]['held_forks'] += 1
                if row[8]:  # right_fork_id
                    state_features[state_id]['held_forks'] += 1
                    
            elif node_type == 'Fork':
                state_features[state_id]['fork_count'] += 1
                if node_state == 'available':
                    state_features[state_id]['available_forks'] += 1
                else:
                    state_features[state_id]['held_forks'] += 1
        
        # Convert to DataFrame
        df = pd.DataFrame(list(state_features.values()))
        
        # Create additional features
        df['total_entities'] = df['philosopher_count'] + df['fork_count']
        df['philosopher_fork_ratio'] = df['philosopher_count'] / df['fork_count']
        df['thinking_ratio'] = df['thinking_philosophers'] / df['philosopher_count']
        df['hungry_ratio'] = df['hungry_philosophers'] / df['philosopher_count']
        df['fork_utilization'] = df['held_forks'] / df['fork_count']
        
        # Add deadlock-related features
        df['deadlock_risk'] = df['hungry_philosophers'] * df['held_forks'] / df['philosopher_count']
        df['resource_contention'] = df['hungry_philosophers'] / (df['available_forks'] + 1)  # +1 to avoid division by zero
        df['deadlock_probability'] = df['is_deadlock'].astype(int)
        
        return df
    
    def create_transition_features(self) -> pd.DataFrame:
        """Create features for transition prediction."""
        query = '''
            SELECT 
                t.from_state,
                t.to_state,
                t.action,
                t.transition_type,
                s1.is_start_state as from_is_start,
                s1.is_final_state as from_is_final,
                s1.has_reach_property as from_has_reach,
                s2.is_start_state as to_is_start,
                s2.is_final_state as to_is_final,
                s2.has_reach_property as to_has_reach
            FROM transitions t
            JOIN states s1 ON t.from_state = s1.state_id
            JOIN states s2 ON t.to_state = s2.state_id
        '''
        
        self.db.cursor.execute(query)
        results = self.db.cursor.fetchall()
        
        transitions = []
        for row in results:
            transitions.append({
                'from_state': row[0],
                'to_state': row[1],
                'action': row[2],
                'transition_type': row[3],
                'from_is_start': row[4],
                'from_is_final': row[5],
                'from_has_reach': row[6],
                'to_is_start': row[7],
                'to_is_final': row[8],
                'to_has_reach': row[9]
            })
        
        return pd.DataFrame(transitions)
    
    def create_sequence_data(self) -> Tuple[List[List[str]], List[str]]:
        """Create sequence data for sequence prediction tasks."""
        # Get all paths through the system
        query = '''
            WITH RECURSIVE paths AS (
                SELECT from_state, to_state, action, 
                       CAST(from_state || '->' || to_state AS TEXT) as path, 
                       1 as depth
                FROM transitions 
                WHERE from_state = 's0'
                
                UNION ALL
                
                SELECT t.from_state, t.to_state, t.action, 
                       p.path || '->' || t.to_state, p.depth + 1
                FROM transitions t
                JOIN paths p ON t.from_state = p.to_state
                WHERE p.depth < 10  -- Limit path length
            )
            SELECT path, action FROM paths
            ORDER BY depth, path
        '''
        
        self.db.cursor.execute(query)
        results = self.db.cursor.fetchall()
        
        sequences = []
        actions = []
        
        for row in results:
            path = row[0].split('->')
            action = row[1]
            sequences.append(path)
            actions.append(action)
        
        return sequences, actions
    
    def create_graph_features(self) -> Dict[str, Any]:
        """Create graph-based features for graph neural networks."""
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes (states)
        query = "SELECT state_id, is_start_state, is_final_state, has_reach_property FROM states"
        self.db.cursor.execute(query)
        states = self.db.cursor.fetchall()
        
        for state in states:
            G.add_node(state[0], 
                      is_start=state[1], 
                      is_final=state[2], 
                      has_reach=state[3])
        
        # Add edges (transitions)
        query = "SELECT from_state, to_state, action FROM transitions"
        self.db.cursor.execute(query)
        transitions = self.db.cursor.fetchall()
        
        for trans in transitions:
            G.add_edge(trans[0], trans[1], action=trans[2])
        
        # Calculate graph features
        graph_features = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'diameter': nx.diameter(G) if nx.is_strongly_connected(G) else None,
            'avg_clustering': nx.average_clustering(G),
            'avg_shortest_path': nx.average_shortest_path_length(G) if nx.is_strongly_connected(G) else None,
            'num_strongly_connected_components': nx.number_strongly_connected_components(G),
            'num_weakly_connected_components': nx.number_weakly_connected_components(G)
        }
        
        return {
            'graph': G,
            'features': graph_features,
            'node_features': nx.get_node_attributes(G, 'is_start'),
            'edge_features': nx.get_edge_attributes(G, 'action')
        }
    
    def prepare_classification_data(self, target_column: str = 'has_reach_property') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for classification tasks."""
        df = self.create_state_features()
        
        # Select features and target
        feature_columns = [
            'philosopher_count', 'fork_count', 'thinking_philosophers', 
            'hungry_philosophers', 'available_forks', 'held_forks',
            'total_entities', 'philosopher_fork_ratio', 'thinking_ratio',
            'hungry_ratio', 'fork_utilization', 'deadlock_risk', 'resource_contention'
        ]
        
        X = df[feature_columns].values
        y = df[target_column].values
        feature_names = feature_columns
        
        # Encode categorical features if any
        for col in df.columns:
            if df[col].dtype == 'object' and col in feature_columns:
                le = LabelEncoder()
                col_idx = feature_columns.index(col)
                X[:, col_idx] = le.fit_transform(X[:, col_idx])
                self.label_encoders[col] = le
        
        return X, y, feature_names
    
    def prepare_deadlock_classification_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data specifically for deadlock classification."""
        df = self.create_state_features()
        
        # Select features for deadlock prediction
        feature_columns = [
            'philosopher_count', 'fork_count', 'thinking_philosophers', 
            'hungry_philosophers', 'available_forks', 'held_forks',
            'total_entities', 'philosopher_fork_ratio', 'thinking_ratio',
            'hungry_ratio', 'fork_utilization', 'deadlock_risk', 'resource_contention'
        ]
        
        X = df[feature_columns].values
        y = df['is_deadlock'].values
        feature_names = feature_columns
        
        return X, y, feature_names
    
    def prepare_regression_data(self, target_column: str = 'fork_utilization') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for regression tasks."""
        df = self.create_state_features()
        
        # Select features and target
        feature_columns = [
            'philosopher_count', 'fork_count', 'thinking_philosophers', 
            'hungry_philosophers', 'available_forks', 'held_forks',
            'total_entities', 'philosopher_fork_ratio', 'thinking_ratio',
            'hungry_ratio'
        ]
        
        X = df[feature_columns].values
        y = df[target_column].values
        feature_names = feature_columns
        
        return X, y, feature_names
    
    def prepare_sequence_data(self) -> Tuple[List[List[int]], List[int]]:
        """Prepare sequence data for RNN/LSTM models."""
        sequences, actions = self.create_sequence_data()
        
        # Create vocabulary
        all_states = set()
        all_actions = set()
        
        for seq in sequences:
            all_states.update(seq)
        all_actions.update(actions)
        
        # Create encoders
        state_encoder = LabelEncoder()
        action_encoder = LabelEncoder()
        
        state_encoder.fit(list(all_states))
        action_encoder.fit(list(all_actions))
        
        # Encode sequences
        encoded_sequences = []
        for seq in sequences:
            encoded_sequences.append(state_encoder.transform(seq))
        
        encoded_actions = action_encoder.transform(actions)
        
        return encoded_sequences, encoded_actions
    
    def save_ml_data(self, output_dir: str = "ml_data"):
        """Save all prepared data for machine learning."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save state features
        state_df = self.create_state_features()
        state_df.to_csv(f"{output_dir}/state_features.csv", index=False)
        
        # Save transition features
        trans_df = self.create_transition_features()
        trans_df.to_csv(f"{output_dir}/transition_features.csv", index=False)
        
        # Save classification data
        X_clf, y_clf, feature_names = self.prepare_classification_data()
        np.save(f"{output_dir}/X_classification.npy", X_clf)
        np.save(f"{output_dir}/y_classification.npy", y_clf)
        
        # Save deadlock classification data
        X_deadlock, y_deadlock, feature_names = self.prepare_deadlock_classification_data()
        np.save(f"{output_dir}/X_deadlock_classification.npy", X_deadlock)
        np.save(f"{output_dir}/y_deadlock_classification.npy", y_deadlock)
        
        # Save regression data
        X_reg, y_reg, feature_names = self.prepare_regression_data()
        np.save(f"{output_dir}/X_regression.npy", X_reg)
        np.save(f"{output_dir}/y_regression.npy", y_reg)
        
        # Save sequence data
        sequences, actions = self.prepare_sequence_data()
        with open(f"{output_dir}/sequences.pkl", 'wb') as f:
            pickle.dump(sequences, f)
        with open(f"{output_dir}/actions.pkl", 'wb') as f:
            pickle.dump(actions, f)
        
        # Save graph data
        graph_data = self.create_graph_features()
        with open(f"{output_dir}/graph_data.pkl", 'wb') as f:
            pickle.dump(graph_data, f)
        
        # Save metadata
        metadata = {
            'feature_names': feature_names,
            'label_encoders': self.label_encoders,
            'num_states': len(state_df),
            'num_transitions': len(trans_df),
            'num_sequences': len(sequences)
        }
        
        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"All ML data saved to {output_dir}/")
        print(f"Files created:")
        print(f"- state_features.csv")
        print(f"- transition_features.csv")
        print(f"- X_classification.npy, y_classification.npy")
        print(f"- X_regression.npy, y_regression.npy")
        print(f"- sequences.pkl, actions.pkl")
        print(f"- graph_data.pkl")
        print(f"- metadata.json")
    
    def close(self):
        """Close the database connection."""
        self.db.close()

def main():
    """Main function to prepare ML data."""
    print("Preparing Dining Philosopher Data for Machine Learning...")
    
    # Create ML data preparation instance
    ml_prep = MLDataPreparation()
    
    # Save all prepared data
    ml_prep.save_ml_data()
    
    # Print summary
    print("\n=== ML DATA SUMMARY ===")
    
    # State features summary
    state_df = ml_prep.create_state_features()
    print(f"State Features: {state_df.shape[0]} states, {state_df.shape[1]} features")
    
    # Transition features summary
    trans_df = ml_prep.create_transition_features()
    print(f"Transition Features: {trans_df.shape[0]} transitions, {trans_df.shape[1]} features")
    
    # Sequence data summary
    sequences, actions = ml_prep.create_sequence_data()
    print(f"Sequence Data: {len(sequences)} sequences, {len(set(actions))} unique actions")
    
    # Graph data summary
    graph_data = ml_prep.create_graph_features()
    print(f"Graph Data: {graph_data['features']['num_nodes']} nodes, {graph_data['features']['num_edges']} edges")
    
    ml_prep.close()
    print("\nML data preparation completed successfully!")

if __name__ == "__main__":
    main() 