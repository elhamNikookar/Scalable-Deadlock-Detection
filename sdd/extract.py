import networkx as nx
import pandas as pd
import numpy as np

dot_file = "phil_15-20-40-100-gts.dot"

try:
    # Try reading the DOT file using pydot backend
    G = nx.drawing.nx_pydot.read_dot(dot_file)
except Exception as e:
    # If that fails, try reading with pygraphviz backend
    import warnings
    warnings.warn("pydot failed, trying pygraphviz...")
    G = nx.drawing.nx_agraph.read_dot(dot_file)

# Display basic graph information
print(f"Number of states (nodes): {G.number_of_nodes()}")
print(f"Number of transitions (edges): {G.number_of_edges()}")

# Show sample of node labels (first 5 states)
sample_states = list(G.nodes(data=True))[:5]
print("Sample states and their attributes:")
for n, d in sample_states:
    print(n, d)

# For deep learning: Convert each state label into a numeric feature vector
state_vectors = []
state_ids = []

for node, attr in G.nodes(data=True):
    label = attr.get('label', '')
    # Remove extra quotes, whitespace, or artifacts common in DOT export
    clean = label.replace('"', '').replace(' ', '').strip()
    # Map characters T/H/E to 0/1/2 respectively for neural network input
    vec = [ {"T":0, "H":1, "E":2}.get(c, -1) for c in clean ]
    state_vectors.append(vec)
    state_ids.append(node)

# Build a dataframe: each row is a state, its ID and vectorized philosopher states
df = pd.DataFrame(state_vectors)
df.insert(0, "state_id", state_ids)

print("First five rows of processed data:")
print(df.head())

# Save to CSV for use in deep learning frameworks (TensorFlow, PyTorch, etc.)
df.to_csv("philosophers_states.csv", index=False)
print("File 'philosophers_states.csv' saved.")

# Optional: Save the adjacency matrix for graph neural network (GNN) applications
adj_mat = nx.to_numpy_array(G, nodelist=state_ids)
np.save("philosophers_adj.npy", adj_mat)
print("Adjacency matrix saved as philosophers_adj.npy.")
