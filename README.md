This is a scipt that visualizes a dense neural network offering full control of
- The colors of nodes and links
- Annotations for input and output nodes
- where the vdots will be placed

# Examples
The code below
```python
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
```

prints
![neural_net_drawing](https://github.com/user-attachments/assets/0d83c9db-3ccf-4910-9bfd-50beeaae2131)


The annotation parameters are written in `select_annotation_parameters`. Specifically:


VDOTS are written using a dictionary `vdots_input_index[(layer_idx, node_idx)]`
```python
(1, 3): vdots
(1, 4): vdots 
(1, 5): vdots 
(1, 6): vdots 
(1, 7): vdots 
(1, 8): vdots 
(1, 9): vdots 
(1, 10): vdots
(1, 11): vdots
(1, 12): vdots
(1, 13): vdots
(2, 3): vdots
(2, 4): vdots
(2, 5): vdots
(2, 6): vdots
(2, 7): vdots
(2, 8): vdots
(2, 9): vdots
(2, 10): vdots
(2, 11): vdots
(2, 12): vdots
(2, 13): vdots
(3, 3): vdots
(3, 4): vdots
(3, 5): vdots
(3, 6): vdots
(3, 7): vdots
(3, 8): vdots
(3, 9): vdots
(3, 10): vdots
(3, 11): vdots
(3, 12): vdots
(3, 13): vdots
(4, 3): vdots
(4, 4): vdots
(4, 5): vdots
(4, 6): vdots
(4, 7): vdots
(4, 8): vdots
(4, 9): vdots
(4, 10): vdots
(4, 11): vdots
(4, 12): vdots
(4, 13): vdots
(5, 3): vdots
(5, 4): vdots
(5, 5): vdots
(5, 6): vdots
(5, 7): vdots
(5, 8): vdots
(5, 9): vdots
(5, 10): vdots
(5, 11): vdots
(5, 12): vdots
(5, 13): vdots
```
