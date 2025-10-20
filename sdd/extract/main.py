import os
import xml.etree.ElementTree as ET
import csv

def extract_philosopher_states(gst_file):
    tree = ET.parse(gst_file)
    root = tree.getroot()
    graph = root.find(".//{*}graph")
    edges = graph.findall("{*}edge")

    # Step 1: Identify philosopher nodes (with self-loop labeled "Phil")
    phil_nodes = set()
    for edge in edges:
        if edge.attrib['from'] == edge.attrib['to']:
            label_attr = edge.find('{*}attr[@name="label"]')
            if (label_attr is not None and
                label_attr.find("{*}string") is not None and
                label_attr.find("{*}string").text == "Phil"):
                phil_nodes.add(edge.attrib['from'])

    # Step 2: For each philosopher node (in sorted order), find their state (another self-loop label)
    states = []
    for node in sorted(phil_nodes, key=lambda x: int(x[1:])):
        state = "unknown"
        for edge in edges:
            if edge.attrib['from'] == node and edge.attrib['to'] == node:
                label_attr = edge.find('{*}attr[@name="label"]')
                if (label_attr is not None and
                    label_attr.find("{*}string") is not None):
                    label = label_attr.find("{*}string").text.strip()
                    if label != "Phil":
                        state = label
        states.append(state)
    return states

# Gather all .gst files in the current directory
gst_files = sorted([f for f in os.listdir('.') if f.endswith('.gst')])

if not gst_files:
    print("No .gst files found in the current directory!")
    exit(1)

# Extract states for each file and build a CSV
rows = []
for gst_file in gst_files:
    states = extract_philosopher_states(gst_file)
    rows.append(states + [gst_file])

# Optional: Print the CSV header (Phil_0, Phil_1, ..., filename)
header = [f'Phil_{i}_state' for i in range(len(rows[0])-1)] + ['filename']

# Write to CSV
output_csv = 'all_gst_states.csv'
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"Extracted states for {len(rows)} files into '{output_csv}'")
