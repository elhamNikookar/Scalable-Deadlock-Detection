import matplotlib.pyplot as plt
import networkx as nx
import random
from matplotlib.patches import FancyArrowPatch

def draw_large_rag(ax, num_proc=10, num_res=10, extra_edges=34,
                   deadlock_cycle=True, title='',
                   highlight_cycle_nodes=True, show_legend=True):
    G = nx.DiGraph()
    processes = [f'P{i+1}' for i in range(num_proc)]
    resources = [f'R{j+1}' for j in range(num_res)]
    G.add_nodes_from(processes + resources)
    xgap = 2.0
    pos = {p: (i*xgap, 2.5) for i, p in enumerate(processes)}
    pos.update({r: (i*xgap, 0) for i, r in enumerate(resources)})

    # Main construction
    cycle_edges, cycle_nodes = [], []
    if deadlock_cycle:
        # Create a multi-node cycle (true deadlock)
        for i in range(num_proc):
            p = processes[i]
            r = resources[i]
            r_req = resources[(i+1)%num_res]
            G.add_edge(p, r_req)
            G.add_edge(r, p)
            cycle_edges.append((p, r_req))
            cycle_edges.append((r, p))
            cycle_nodes += [p, r_req, r]
    else:
        # BENIGN: Assign Ri to Pi, but requests are to a DIFFERENT resource (no trivial 2-cycles)
        for i in range(num_proc):
            p = processes[i]
            r = resources[i]
            G.add_edge(r, p)  # assignment: Ri held by Pi

        for i in range(num_proc):
            p = processes[i]
            r_req = resources[(i+2)%num_res]  # always a different resource
            if (r_req, p) not in G.edges():
                G.add_edge(p, r_req)  # Pi requests a resource not held by itself

        cycle_edges, cycle_nodes = [], []

    # Other random edges
    edge_set = set(G.edges())
    for _ in range(extra_edges):
        # Only add if this does not create a 2-way edge for benign
        p, r = random.choice(processes), random.choice(resources)
        if (p, r) not in edge_set and (not (not deadlock_cycle and (r,p) in edge_set)):
            G.add_edge(p, r)
            edge_set.add((p, r))
        p2, r2 = random.choice(processes), random.choice(resources)
        if (r2, p2) not in edge_set and (not (not deadlock_cycle and (p2,r2) in edge_set)):
            G.add_edge(r2, p2)
            edge_set.add((r2, p2))

    # Node visuals
    node_color_map = []
    node_edgecolors = []
    for n in G.nodes():
        color = "cornflowerblue" if n.startswith("P") else "orange"
        node_color_map.append(color)
        if highlight_cycle_nodes and (n in cycle_nodes):
            node_edgecolors.append('crimson')
        else:
            node_edgecolors.append('black')
    node_shapes = ['o' if n.startswith("P") else 's' for n in G.nodes()]
    shape_map = {n: ('o' if n.startswith("P") else 's') for n in G.nodes()}

    # Draw nodes by type
    for shape in set(node_shapes):
        nlist = [n for n in G.nodes() if shape_map[n] == shape]
        idxs = [list(G.nodes()).index(n) for n in nlist]
        edgeclrs = [node_edgecolors[i] for i in idxs]
        nx.draw_networkx_nodes(
            G, pos, nodelist=nlist,
            node_shape=shape, node_size=480,
            node_color=[node_color_map[i] for i in idxs],
            edgecolors=edgeclrs,
            linewidths=2.2 if highlight_cycle_nodes else 1.5,
            ax=ax
        )

    # Draw non-cycle edges regularly
    non_cycle = [e for e in G.edges() if e not in cycle_edges]
    nx.draw_networkx_edges(
        G, pos, edgelist=non_cycle, edge_color="dimgray",
        style="solid", width=1.2, ax=ax, arrows=True,
        arrowsize=13, alpha=0.30, arrowstyle='-|>'
    )

    # Draw cycle edges (non-bold, arrow at path end, curved)
    if deadlock_cycle and cycle_edges:
        for (src, dst) in cycle_edges:
            src_xy = pos[src]
            dst_xy = pos[dst]
            is_proc = src.startswith("P")
            rad = 0.22 if is_proc else -0.22
            fa = FancyArrowPatch(
                src_xy, dst_xy,
                arrowstyle='-|>',
                color="crimson",
                linewidth=2.3,
                connectionstyle=f"arc3,rad={rad}",
                mutation_scale=21,
                alpha=0.95,
                shrinkA=14,
                shrinkB=20
            )
            ax.add_patch(fa)

    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', ax=ax, font_color='black')

    # --- HORIZONTAL LEGEND BELOW PLOT ---
    if show_legend:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='crimson', ls='--', lw=2.3, label='Deadlock Cycle Edge',
                   marker=None),
            Line2D([0], [0], color='dimgray', ls='-', lw=1.2, label='Other Edge'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='cornflowerblue',
                   markeredgecolor='crimson' if highlight_cycle_nodes else 'black',
                   markersize=11, label='Process (in cycle)' if highlight_cycle_nodes else 'Process'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='cornflowerblue',
                   markeredgecolor='black',
                   markersize=11, label='Process (not in cycle)'),
            Line2D([0],[0], marker='s', color='w', markerfacecolor='orange',
                   markeredgecolor='crimson' if highlight_cycle_nodes else 'black',
                   markersize=11, label='Resource (in cycle)' if highlight_cycle_nodes else 'Resource'),
            Line2D([0],[0], marker='s', color='w', markerfacecolor='orange',
                   markeredgecolor='black',
                   markersize=11, label='Resource (not in cycle)'),
        ]
        custom_leg = ax.legend(
            handles=legend_elements,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.26),
            fontsize=10,
            frameon=True,
            borderaxespad=0.4,
            ncol=6,
            title="Legend",
            title_fontsize=11
        )
        custom_leg.get_frame().set_alpha(0.94)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=16, pad=17, weight="bold")
    ax.axis("off")
    ax.set_aspect('equal')

# ======= Generate Example Figures ============
for is_deadlock, fname, title in [
    (True, "rag_deadlock_10x10-legend.pdf", "Deadlock: Correctly Identified (10×10)"),
    (False, "rag_benign_10x10-legend.pdf", "Benign State: Falsely Labeled (10×10)")
]:
    fig, ax = plt.subplots(figsize=(20, 6))
    draw_large_rag(
        ax, num_proc=10, num_res=10, extra_edges=40,
        deadlock_cycle=is_deadlock,
        title=title, highlight_cycle_nodes=True, show_legend=True
    )
    ax.set_title(title, fontsize=28)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(20)
        legend.set_bbox_to_anchor((0.5, -0.18))  # Move legend lower if legend already exists outside draw_large_rag
        legend._loc = 9  # 'upper center'
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.23)
    plt.savefig(fname, bbox_inches='tight')
    plt.savefig(fname.replace('.pdf', '.png'), dpi=430, bbox_inches='tight')
    plt.show()
