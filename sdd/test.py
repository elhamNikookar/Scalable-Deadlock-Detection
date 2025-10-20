import xml.etree.ElementTree as ET
import pandas as pd

# === 1. Parse GXL file ===
gxl_file = 'phil_15-20-40-100@start-2.gxl'
tree = ET.parse(gxl_file)
root = tree.getroot()
ns = {'gxl': 'http://www.gupro.de/GXL/gxl-1.0.dtd'}

graph = root.find('gxl:graph', ns)

# === 2. Extract nodes ===
nodes = []
for node in graph.findall('gxl:node', ns):
    node_id = node.attrib['id']
    # Try to extract state info (adapt as needed)
    state = None
    for attr in node.findall('gxl:attr', ns):
        if attr.attrib['name'].lower() == 'state':
            state_str = attr.find('gxl:string', ns)
            if state_str is not None:
                state = state_str.text
    nodes.append({'node_id': node_id, 'state': state})

# Save nodes to CSV
df_nodes = pd.DataFrame(nodes)
df_nodes.to_csv('gxl_nodes.csv', index=False)
print("Nodes saved to gxl_nodes.csv")
print(df_nodes.head())

# === 3. Extract edges ===
edges = []
for edge in graph.findall('gxl:edge', ns):
    src = edge.attrib['from']
    tgt = edge.attrib['to']
    label = None
    for attr in edge.findall('gxl:attr', ns):
        if attr.attrib['name'].lower() == 'label':
            label_str = attr.find('gxl:string', ns)
            if label_str is not None:
                label = label_str.text
    edges.append({'from': src, 'to': tgt, 'label': label})

# Save edges to CSV
df_edges = pd.DataFrame(edges)
df_edges.to_csv('gxl_edges.csv', index=False)
print("Edges saved to gxl_edges.csv")
print(df_edges.head())
