import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import numpy as np

def draw_neural_net(ax, layer_sizes, real_layers, dimensions, input_colors=None, output_colors = None, vdots_input_index=None, input_arrows=None, output_arrows=None, color_explanations=None):
    v_spacing, h_spacing, radius, start_x_arrow, offset_x_arrowhead = dimensions

    n_layers = len(layer_sizes)
    node_coords = {}  # (layer, node_idx) -> (x, y, is_node)

    # Preprocess vdots into a lookup
    vdots_input_index = vdots_input_index or {}
    vdots_lookup = {
        (lidx, nidx): position for (lidx, nidx), position in vdots_input_index.items()
    }

    # Draw nodes and vdots in unified loop
    max_top_y = -100
    for layer_idx, n_nodes in enumerate(layer_sizes):
        layer_x = layer_idx * h_spacing
        top_y = (n_nodes - 1) * v_spacing / 2
        if top_y > max_top_y:
            max_top_y = top_y
            
        for node_idx in range(n_nodes):
            y = top_y - node_idx * v_spacing
            coord_key = (layer_idx, node_idx)

            if coord_key in vdots_lookup:
                # Treat as a vdot (replace the node)
                node_coords[(layer_idx, node_idx)] = (layer_x, y, 0)  # Mark as vdot (is_node = 0)
                ax.text(layer_x, y, r'$\vdots$', ha='center', va='center', fontsize=16)
            else:
                # Draw actual node
                node_coords[(layer_idx, node_idx)] = (layer_x, y, 1)  # Mark as real node (is_node = 1)

                # Choose color
                if layer_idx == 0 and input_colors:
                    color = input_colors[node_idx % len(input_colors)]
                elif layer_idx == 0:
                    color = 'lightblue'
                elif layer_idx == n_layers - 1 and output_colors:
                    color = output_colors[node_idx % len(output_colors)]
                elif layer_idx == n_layers - 1:
                    color = 'salmon'
                else:
                    color = 'lightgray'

                circ = Circle((layer_x, y), radius=radius, color=color, ec='black', zorder=3)
                ax.add_patch(circ)

    for layer_idx, n_nodes in enumerate(real_layers):
        ax.text(layer_idx* h_spacing, max_top_y+10, f'$n_{{nodes}}$={n_nodes}', ha='center', va='bottom', fontsize=12, color='black', rotation=90)


    # Draw connections between real nodes only
    for layer_idx in range(n_layers - 1):
        for i in range(layer_sizes[layer_idx]):
            for j in range(layer_sizes[layer_idx + 1]):
                coord_i = node_coords.get((layer_idx, i))
                coord_j = node_coords.get((layer_idx + 1, j))
                if coord_i and coord_j:
                    x1, y1, is_node_i = coord_i
                    x2, y2, is_node_j = coord_j
                    if is_node_i and is_node_j:
                        line = Line2D([x1, x2], [y1, y2], color='gray', linewidth=1)
                        ax.add_line(line)

    # Draw input arrows with optional text labels
    if input_arrows:
        top_y = (layer_sizes[0] - 1) * v_spacing / 2
        for lab, (layer, end, label) in input_arrows.items():
            # Start position is outside the network, to the left of the first layer
            start_y = top_y - (end) * v_spacing  # Get the Y position of the target node in the input layer
            x_arrow_marker = 0+offset_x_arrowhead
            
            
            # Draw the line (no marker at the start)
            arrow = Line2D([start_x_arrow, x_arrow_marker], [start_y, start_y], color='black', linewidth=2, linestyle='-')
            ax.add_line(arrow)
            
            # Add the arrowhead marker at the end
            ax.plot(x_arrow_marker, start_y, marker='>', color='black', markersize=10)  # Arrowhead at the end
            
            # Optional text label above the arrow
            ax.text((start_x_arrow + x_arrow_marker) / 2, start_y - 0, label, ha='center', va='bottom', fontsize=12, color='black')

    # Draw output arrows with optional text labels
    if output_arrows:
        top_y = (layer_sizes[-1] - 1) * v_spacing / 2
        for lab, (layer, start_idx, label) in output_arrows.items():
            start_x = (n_layers - 1) * h_spacing
            start_y = top_y - start_idx * v_spacing
            end_x = start_x + abs(start_x_arrow)  # Same length as input arrows
            arrow = Line2D([start_x, end_x], [start_y, start_y], color='black', linewidth=2, linestyle='-')
            ax.add_line(arrow)
            ax.plot(end_x, start_y, marker='>', color='black', markersize=10)
            ax.text((start_x + end_x) / 2, start_y - 0, label, ha='center', va='bottom', fontsize=12, color='black')


    if color_explanations:
        fontsize = 10
        displacement = fontsize-4
        unique_colors_used_in_nodes = np.unique(input_colors+output_colors)
        z=0
        for color in color_explanations:
            if color[0] in unique_colors_used_in_nodes:
                ax.text(-4, -max_top_y - 3*radius - displacement*z, color[0] + ": " + color[1], ha='left', va='bottom', fontsize=fontsize, color=color[0])
                z+=1

    ax.set_aspect('equal')
    ax.axis('off')
    ax.autoscale_view()
    
    
def select_annotation_parameters(sim=None):
    
    try:
        color_explanations = [
            ['red',  'scaled using StandardScaler()'], 
            ['blue', 'Scaled Between [-1, 1]']
            ]
        if sim.problem_data.get_only_diagnostic_output_from_forward and sim.problem_data.use_variable_q_in and sim.problem_data.mode_13:
        # Define arrows with optional labels (arrows from outside the network to input nodes)
            input_arrows = {
            (0, 0): (0, 0, "$p_{p}^{(1)}$"),
            (0, 1): (0, 1, "$p_{p}^{(2)}$"),
            (0, 3): (0, 2, "$p_{p}^{(3)}$"),
            (0, 4): (0, 3, "$q_{r}^{(1)}$"),
            (0, 5): (0, 4, "$q_{r}^{(2)}$"),
            (0, 6): (0, 5, "$q_{r}^{(3)}$"),
            (0, 7): (0, 6, "$q_{p}^{(1)}$"),
            (0, 8): (0, 7, "$q_{p}^{(2)}$"),
            (0, 9): (0, 8, "$q_{p}^{(3)}$"),
            (0, 10): (0, 9, "$L$"),
            }
            
            L = len(input_arrows.keys())
            input_node_colors = ['red' for _ in range(L)]
            input_node_colors[-1] = 'blue'
            
            for i, par in enumerate(sim.problem_data.pars_perturb):
                input_arrows[(0, L+i+1)] = (0, L+i, f"${par}$")
                input_node_colors + ['red']

            output_arrows = {
            (5, 0): (5, 0, "$d_{1}^{(1)}$"),
            (5, 1): (5, 1, "$d_{1}^{(2)}$"),
            (5, 2): (5, 2, "$d_{2}^{(1)}$"),
            (5, 3): (5, 3, "$d_{2}^{(2)}$"),
            (5, 4): (5, 4, "$d_{3}^{(1)}$"),
            (5, 5): (5, 5, "$d_{3}^{(2)}$"),
            }
        
            output_node_colors = ['blue' for _ in range(len(output_arrows))]
            
            
        
        elif sim.problem_data.get_only_diagnostic_output_from_forward and sim.problem_data.mode_13:
            
            input_arrows = {
            (0, 0): (0, 0, "$p_{p}^{(1)}$"),
            (0, 1): (0, 1, "$p_{p}^{(2)}$"),
            (0, 3): (0, 2, "$p_{p}^{(3)}$"),
            (0, 4): (0, 3, "$q_{r}^{(1)}$"),
            (0, 5): (0, 4, "$q_{r}^{(2)}$"),
            (0, 6): (0, 5, "$q_{r}^{(3)}$"),
            (0, 7): (0, 6, "$L$"),
            }
            
            L = len(input_arrows.keys())
            input_node_colors = ['red' for _ in range(L)]
            input_node_colors[-1] = 'blue'
            
            for i, par in enumerate(sim.problem_data.pars_perturb):
                input_arrows[(0, L+i+1)] = (0, L+i, f"${par}$")
                input_node_colors + ['red']

            output_arrows = {
            (5, 0): (5, 0, "$d_{1}^{(1)}$"),
            (5, 1): (5, 1, "$d_{1}^{(2)}$"),
            (5, 2): (5, 2, "$d_{2}^{(1)}$"),
            (5, 3): (5, 3, "$d_{2}^{(2)}$"),
            (5, 4): (5, 4, "$d_{3}^{(1)}$"),
            (5, 5): (5, 5, "$d_{3}^{(2)}$"),
            }
        
            output_node_colors = ['blue' for _ in range(len(output_arrows))]

        
        else:
            raise NotImplementedError
    except:
        input_arrows = {
            (0, 0): (0, 0, "$p_{p}^{(1)}$"),
            (0, 1): (0, 1, "$p_{p}^{(2)}$"),
            (0, 3): (0, 2, "$p_{p}^{(3)}$"),
            (0, 4): (0, 3, "$q_{r}^{(1)}$"),
            (0, 5): (0, 4, "$q_{r}^{(2)}$"),
            (0, 6): (0, 5, "$q_{r}^{(3)}$"),
            (0, 7): (0, 6, "$q_{p}^{(1)}$"),
            (0, 8): (0, 7, "$q_{p}^{(2)}$"),
            (0, 9): (0, 8, "$q_{p}^{(3)}$"),
            (0, 10): (0, 9, "$L$"),
            }
        
        L = len(input_arrows.keys())
        input_node_colors = ['red' for _ in range(L)]
        input_node_colors[-1] = 'blue'

        output_arrows = {
        (5, 0): (5, 0, "$d_{1}^{(1)}$"),
        (5, 1): (5, 1, "$d_{1}^{(2)}$"),
        (5, 2): (5, 2, "$d_{2}^{(1)}$"),
        (5, 3): (5, 3, "$d_{2}^{(2)}$"),
        (5, 4): (5, 4, "$d_{3}^{(1)}$"),
        (5, 5): (5, 5, "$d_{3}^{(2)}$"),
        }
    
        output_node_colors = ['blue' for _ in range(len(output_arrows))]

    color_explanations = [
        ['red',  'Explanation for red nodes'], 
        ['blue', 'Explanation for blue nodes']
        ]
    
    return input_arrows, output_arrows, input_node_colors, output_node_colors, color_explanations





# Example usage

real_layers = [12, 1500, 1500, 1500, 1500, 1500, 6]
layers = [real_layers[0], 15, 15, 15, 15, 15, real_layers[-1]]

vdots_input_index = {
    # (5, 3): 'vdots',
    # (5, 4): 'vdots',
    # (5, 5): 'vdots',
    # (5, 6): 'vdots',
}

hidden_layer_nodes_ommit = [0, 1, 2, -1]

for i in range(len(layers)-2):
    j = i + 1
    nodes = np.array([x for x in range(layers[j])])
    for node in nodes:
        if not node in nodes[hidden_layer_nodes_ommit]:
            vdots_input_index[(j, node)] = 'vdots'


# input_arrows = None
v_spacing = 12.2
h_spacing = 19.0
radius = 5.65
start_x_arrow = -30
offset_x_arrowhead = -9
dimensions = (v_spacing, h_spacing, radius, start_x_arrow, offset_x_arrowhead)
fig, ax = plt.subplots(figsize=(10, 6))

input_arrows, output_arrows, input_colors, output_colors, color_explanations = select_annotation_parameters()


draw_neural_net(
    ax,
    layer_sizes=layers,
    real_layers=real_layers,
    dimensions=dimensions,
    input_colors=input_colors,
    output_colors=output_colors,
    vdots_input_index=vdots_input_index,
    input_arrows=input_arrows,  # Pass the input arrows here
    output_arrows=output_arrows,
    color_explanations=color_explanations
)

for extension in ['.jpg', '.pdf']:
    plt.savefig(f"neural_net_drawing{extension}", bbox_inches='tight')
plt.show()
