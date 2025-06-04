import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# --- Activation Functions ---
def step_function(x):
    return np.where(x >= 0, 1, 0)

def sigmoid_function(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500))) # Added clip for stability

def relu_function(x):
    return np.maximum(0, x)

def linear_function(x):
    return x

def tanh_function(x):
    return np.tanh(x)

activation_functions = {
    "Step": step_function,
    "Sigmoid": sigmoid_function,
    "ReLU": relu_function,
    "Tanh": tanh_function,
    "Linear": linear_function
}

# --- MLP Logic ---
class MLP:
    def __init__(self, layer_config, activation_hidden_name="ReLU", activation_output_name="Sigmoid"):
        """
        Initializes the Multi-Layer Perceptron.
        layer_config: List of integers representing the number of neurons in each layer.
                      e.g., [num_inputs, neurons_hidden1, neurons_hidden2, ..., num_outputs]
        activation_hidden_name: Name of the activation function for hidden layers.
        activation_output_name: Name of the activation function for the output layer.
        """
        self.num_layers = len(layer_config)
        self.layer_config = layer_config
        self.activation_hidden = activation_functions.get(activation_hidden_name, relu_function)
        self.activation_output = activation_functions.get(activation_output_name, sigmoid_function)
        
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(self.num_layers - 1):
            # Xavier/Glorot initialization (good for Sigmoid/Tanh)
            # He initialization (good for ReLU)
            limit = np.sqrt(6 / (layer_config[i] + layer_config[i+1]))
            if activation_hidden_name == "ReLU" and i < self.num_layers - 2: # He for hidden ReLU layers
                 w = np.random.randn(layer_config[i], layer_config[i+1]) * np.sqrt(2. / layer_config[i])
            else: # Xavier/Glorot for output or other hidden activations
                 w = np.random.uniform(-limit, limit, (layer_config[i], layer_config[i+1]))

            b = np.zeros((1, layer_config[i+1])) # Initialize biases to zero
            
            self.weights.append(w)
            self.biases.append(b)

    def predict(self, inputs_array):
        """
        Performs forward propagation through the MLP.
        inputs_array: Numpy array of input features.
        Returns the final output and a list of activations for each layer (for visualization).
        """
        if not isinstance(inputs_array, np.ndarray):
            inputs_array = np.array(inputs_array)
        
        if inputs_array.ndim == 1:
            inputs_array = inputs_array.reshape(1, -1) # Ensure it's a row vector

        if inputs_array.shape[1] != self.layer_config[0]:
            st.error(f"Input dimension mismatch. Expected {self.layer_config[0]}, got {inputs_array.shape[1]}")
            return None, []

        activations = [inputs_array] # Store activations of each layer, starting with input
        a = inputs_array

        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1: # Hidden layer
                a = self.activation_hidden(z)
            else: # Output layer
                a = self.activation_output(z)
            activations.append(a)
            
        return a, activations


# --- Plotting Functions ---
def plot_activation_function_graph(activation_name):
    if activation_name not in activation_functions:
        st.error(f"Cannot plot unknown activation function: {activation_name}")
        return

    func = activation_functions[activation_name]
    x_range = np.linspace(-5, 5, 200)
    y_range = func(x_range)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_range, y_range)
    ax.set_title(f"{activation_name} Activation Function", fontsize=10)
    ax.set_xlabel("Input (z)", fontsize=8)
    ax.set_ylabel("Output (φ(z))", fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    st.pyplot(fig)

def create_mlp_diagram(layer_config, layer_activations=None, activation_hidden_name="", activation_output_name=""):
    """Creates a Plotly diagram of the MLP structure."""
    fig = go.Figure()
    
    num_total_layers = len(layer_config)
    max_neurons_in_layer = max(layer_config) if layer_config else 1
    
    node_x = []
    node_y = []
    node_labels = []
    node_colors = []
    node_line_colors = []
    node_sizes = []

    # Layer spacing and node positioning
    layer_x_coords = np.linspace(0, num_total_layers * 2, num_total_layers) # Increased spacing

    for i, num_neurons in enumerate(layer_config):
        # Vertical positioning: center neurons in their layer
        y_start = (num_neurons - 1) / 2.0 
        
        for j in range(num_neurons):
            node_x.append(layer_x_coords[i])
            # Spread neurons out, ensure y_start is adjusted if max_neurons_in_layer is large
            node_y.append(y_start - j) 
            
            label_parts = []
            if i == 0:
                label_parts.append(f"Input {j+1}")
            elif i == num_total_layers - 1:
                label_parts.append(f"Output {j+1}")
                label_parts.append(f"<small>({activation_output_name})</small>")
            else:
                label_parts.append(f"H{i}-N{j+1}") # Hidden Layer i, Neuron j+1
                label_parts.append(f"<small>({activation_hidden_name})</small>")

            # Add activation value if available
            activation_val_str = ""
            if layer_activations and len(layer_activations) > i and layer_activations[i] is not None:
                try:
                    val = layer_activations[i][0, j] # Assuming batch size 1 for display
                    activation_val_str = f"<br><i>Act: {val:.2f}</i>"
                except (IndexError, TypeError):
                    activation_val_str = "<br><i>Act: N/A</i>" # Should not happen if layer_activations is correct
            
            node_labels.append("<br>".join(label_parts) + activation_val_str)
            
            # Color coding
            color = 'rgba(173, 216, 230, 0.7)' # Default: lightblue (input)
            line_color = 'blue'
            if i > 0 and i < num_total_layers - 1: # Hidden layer
                color = 'rgba(144, 238, 144, 0.7)' # lightgreen
                line_color = 'green'
            elif i == num_total_layers - 1: # Output layer
                color = 'rgba(255, 182, 193, 0.7)' # lightcoral
                line_color = 'red'
            
            node_colors.append(color)
            node_line_colors.append(line_color)
            node_sizes.append(18 if max_neurons_in_layer <= 10 else 15) # Adjust size based on density

    # Edges (Connections between layers)
    edge_x_coords = []
    edge_y_coords = []
    neuron_idx_offset = 0
    for i in range(num_total_layers - 1):
        neurons_in_current_layer = layer_config[i]
        neurons_in_next_layer = layer_config[i+1]
        
        for j in range(neurons_in_current_layer):
            current_neuron_global_idx = neuron_idx_offset + j
            for k in range(neurons_in_next_layer):
                next_neuron_global_idx = neuron_idx_offset + neurons_in_current_layer + k
                edge_x_coords.extend([node_x[current_neuron_global_idx], node_x[next_neuron_global_idx], None])
                edge_y_coords.extend([node_y[current_neuron_global_idx], node_y[next_neuron_global_idx], None])
        neuron_idx_offset += neurons_in_current_layer

    fig.add_trace(go.Scatter(
        x=edge_x_coords, y=edge_y_coords,
        mode='lines',
        line=dict(width=0.5, color='rgba(128,128,128,0.3)'), # Lighter, thinner lines for MLP
        hoverinfo='none'
    ))

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[f"<b>{label.split('<br>')[0]}</b><br>{'<br>'.join(label.split('<br>')[1:])}" for label in node_labels],
        textposition="middle right", 
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=1.5, color=node_line_colors)),
        hoverinfo='text',
        textfont=dict(size=8)
    ))
    
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-1, layer_x_coords[-1] + 1 if layer_x_coords.size >0 else 1]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, autorange="reversed", 
                   scaleanchor="x", scaleratio=0.8 if max_neurons_in_layer > 5 else 1), # Adjust aspect ratio
        height=max(400, max_neurons_in_layer * 40 + 100), # Dynamic height
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def plot_decision_boundary_mlp(inputs_data, targets_data, mlp_instance, activation_output_name):
    """Plots data points and the decision boundary for a 2-input MLP."""
    if inputs_data.shape[1] != 2:
        st.warning("Decision boundary plot is only available for 2 inputs.")
        return
    if mlp_instance.layer_config[0] != 2 or mlp_instance.layer_config[-1] != 1:
        st.warning("Decision boundary plot is best for MLPs with 2 inputs and 1 output neuron.")
        return

    fig, ax = plt.subplots(figsize=(6,5))
    
    colors = ['red' if t == 0 else 'blue' for t in targets_data]
    ax.scatter(inputs_data[:, 0], inputs_data[:, 1], c=colors, edgecolors='k', s=50, alpha=0.8, label='Data Points')

    x_min, x_max = inputs_data[:, 0].min() - 0.5, inputs_data[:, 0].max() + 0.5
    y_min, y_max = inputs_data[:, 1].min() - 0.5, inputs_data[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z, _ = mlp_instance.predict(grid_points) # Z will be (N, 1) for 1 output neuron
    
    # For binary classification with sigmoid/step, threshold at 0.5
    if activation_output_name in ["Sigmoid", "Step"]:
        Z = (Z >= 0.5).astype(int)
    
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, levels=np.linspace(Z.min(), Z.max(), 3), cmap=plt.cm.RdYlBu)
    ax.contour(xx, yy, Z, colors='k', linewidths=0.7, levels=[0.5] if activation_output_name in ["Sigmoid", "Step"] else np.linspace(Z.min(), Z.max(), 3)[1:-1])

    ax.set_xlabel("Input 1 (x1)", fontsize=9)
    ax.set_ylabel("Input 2 (x2)", fontsize=9)
    ax.set_title(f"MLP Decision Boundary ({activation_output_name} output)", fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    st.pyplot(fig)

# --- Logic Gate Data ---
gate_data = {
    "AND": {"inputs": np.array([[0,0],[0,1],[1,0],[1,1]]), "outputs": np.array([0,0,0,1]), "num_inputs": 2, "num_outputs": 1},
    "OR":  {"inputs": np.array([[0,0],[0,1],[1,0],[1,1]]), "outputs": np.array([0,1,1,1]), "num_inputs": 2, "num_outputs": 1},
    "NAND":{"inputs": np.array([[0,0],[0,1],[1,0],[1,1]]), "outputs": np.array([1,1,1,0]), "num_inputs": 2, "num_outputs": 1},
    "NOR": {"inputs": np.array([[0,0],[0,1],[1,0],[1,1]]), "outputs": np.array([1,0,0,0]), "num_inputs": 2, "num_outputs": 1},
    "XOR": {"inputs": np.array([[0,0],[0,1],[1,0],[1,1]]), "outputs": np.array([0,1,1,0]), "num_inputs": 2, "num_outputs": 1,
            "note": "XOR is not linearly separable by a single perceptron, but an MLP can solve it (e.g., with a [2,2,1] structure)."}
}

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="MLP Visualizer")

st.sidebar.title("🧠 MLP Controls")

# Initialize session state
default_hidden_layers = 1
default_neurons_per_hidden = [2] # Default for 1 hidden layer

if 'num_inputs' not in st.session_state: st.session_state.num_inputs = 2
if 'num_outputs' not in st.session_state: st.session_state.num_outputs = 1
if 'num_hidden_layers' not in st.session_state: st.session_state.num_hidden_layers = default_hidden_layers
if 'neurons_per_hidden_layer' not in st.session_state:
    st.session_state.neurons_per_hidden_layer = list(default_neurons_per_hidden) # Ensure it's a list
if 'activation_hidden' not in st.session_state: st.session_state.activation_hidden = "ReLU"
if 'activation_output' not in st.session_state: st.session_state.activation_output = "Sigmoid"
if 'user_inputs_values' not in st.session_state:
    st.session_state.user_inputs_values = {f"x{i+1}": 0.0 for i in range(st.session_state.num_inputs)}
if 'layer_activations_cache' not in st.session_state: st.session_state.layer_activations_cache = None
if 'current_mlp_output' not in st.session_state: st.session_state.current_mlp_output = None
if 'run_simulation' not in st.session_state: st.session_state.run_simulation = False
if 'mlp_instance' not in st.session_state: st.session_state.mlp_instance = None


def reset_simulation_state():
    st.session_state.run_simulation = False
    st.session_state.layer_activations_cache = None
    st.session_state.current_mlp_output = None
    st.session_state.mlp_instance = None # Force re-creation of MLP with new architecture/weights

def update_neurons_list_length():
    # Adjust the length of neurons_per_hidden_layer based on num_hidden_layers
    current_len = len(st.session_state.neurons_per_hidden_layer)
    target_len = st.session_state.num_hidden_layers
    if current_len < target_len:
        st.session_state.neurons_per_hidden_layer.extend([2] * (target_len - current_len)) # Default to 2 neurons
    elif current_len > target_len:
        st.session_state.neurons_per_hidden_layer = st.session_state.neurons_per_hidden_layer[:target_len]
    reset_simulation_state()

# --- Sidebar Controls ---
st.sidebar.header("MLP Architecture")
selected_gate = st.sidebar.selectbox(
    "Load Logic Gate Example:", ["Custom"] + list(gate_data.keys()), 
    index=0, key="gate_selector", on_change=reset_simulation_state
)

if selected_gate != "Custom":
    gate_info = gate_data[selected_gate]
    st.session_state.num_inputs = gate_info["num_inputs"]
    st.session_state.num_outputs = gate_info["num_outputs"]
    if "note" in gate_info: st.sidebar.info(gate_info["note"])
    # For XOR, suggest a common MLP structure
    if selected_gate == "XOR":
        st.session_state.num_hidden_layers = 1
        st.session_state.neurons_per_hidden_layer = [2] # [2 inputs, 2 hidden, 1 output]
    st.session_state.num_inputs_disabled = True
    st.session_state.num_outputs_disabled = True
else:
    st.session_state.num_inputs_disabled = False
    st.session_state.num_outputs_disabled = False

st.session_state.num_inputs = st.sidebar.slider(
    "Number of Input Neurons:", 1, 10, value=st.session_state.num_inputs, 
    key="num_inputs_slider", disabled=st.session_state.num_inputs_disabled, on_change=reset_simulation_state
)

st.session_state.num_hidden_layers = st.sidebar.slider(
    "Number of Hidden Layers:", 0, 5, value=st.session_state.num_hidden_layers,
    key="num_hidden_layers_slider", on_change=update_neurons_list_length # This calls reset_simulation_state
)

# Ensure neurons_per_hidden_layer list is correctly sized before rendering inputs for it
if len(st.session_state.neurons_per_hidden_layer) != st.session_state.num_hidden_layers:
    update_neurons_list_length() # Adjust if somehow out of sync (e.g. initial load)

if st.session_state.num_hidden_layers > 0:
    st.sidebar.subheader("Neurons per Hidden Layer")
    temp_neurons_list = list(st.session_state.neurons_per_hidden_layer) # Work on a copy
    for i in range(st.session_state.num_hidden_layers):
        temp_neurons_list[i] = st.sidebar.number_input(
            f"Neurons in Hidden Layer {i+1}:", min_value=1, max_value=20, 
            value=temp_neurons_list[i], key=f"hidden_layer_{i}_neurons",
            on_change=reset_simulation_state
        )
    st.session_state.neurons_per_hidden_layer = temp_neurons_list


st.session_state.num_outputs = st.sidebar.slider(
    "Number of Output Neurons:", 1, 10, value=st.session_state.num_outputs,
    key="num_outputs_slider", disabled=st.session_state.num_outputs_disabled, on_change=reset_simulation_state
)

st.sidebar.markdown("---")
st.sidebar.header("Activation Functions")
st.session_state.activation_hidden = st.sidebar.selectbox(
    "Hidden Layer Activation (φ_h):", list(activation_functions.keys()),
    index=list(activation_functions.keys()).index(st.session_state.activation_hidden),
    key="activation_hidden_selector", on_change=reset_simulation_state
)
st.session_state.activation_output = st.sidebar.selectbox(
    "Output Layer Activation (φ_o):", list(activation_functions.keys()),
    index=list(activation_functions.keys()).index(st.session_state.activation_output),
    key="activation_output_selector", on_change=reset_simulation_state
)


# --- Main Application Area ---
st.title("🧠 Interactive MLP Visualizer")
st.markdown("""
Design and visualize a Multi-Layer Perceptron (MLP). 
Adjust its architecture and activation functions in the sidebar. 
Provide inputs to see how data propagates and the MLP computes the output. 
Weights and biases are initialized when the architecture changes.
""")

# Construct layer configuration
layer_config = [st.session_state.num_inputs] + \
               [n for n in st.session_state.neurons_per_hidden_layer if st.session_state.num_hidden_layers > 0] + \
               [st.session_state.num_outputs]


# Create or get cached MLP instance
if st.session_state.mlp_instance is None or \
   st.session_state.mlp_instance.layer_config != layer_config or \
   st.session_state.mlp_instance.activation_hidden != activation_functions[st.session_state.activation_hidden] or \
   st.session_state.mlp_instance.activation_output != activation_functions[st.session_state.activation_output]:
    
    st.session_state.mlp_instance = MLP(
        layer_config,
        st.session_state.activation_hidden,
        st.session_state.activation_output
    )
    # Clear previous run results if MLP is new
    st.session_state.layer_activations_cache = None 
    st.session_state.current_mlp_output = None
    st.session_state.run_simulation = False


mlp = st.session_state.mlp_instance


col_structure, col_activation_plot = st.columns([3, 2])

with col_structure:
    st.subheader("MLP Structure & Data Flow")
    activations_to_plot = st.session_state.layer_activations_cache if st.session_state.run_simulation else None
    
    # Ensure input layer activations are prepped for the diagram even before run
    if activations_to_plot is None and st.session_state.user_inputs_values:
        initial_input_values = [st.session_state.user_inputs_values.get(f"x{i+1}", 0.0) for i in range(st.session_state.num_inputs)]
        activations_to_plot = [np.array(initial_input_values).reshape(1, -1)] # Initial input layer display
        # Pad with Nones for other layers if not run yet
        activations_to_plot.extend([None] * (len(layer_config) -1))

    mlp_fig = create_mlp_diagram(
        layer_config,
        layer_activations=activations_to_plot,
        activation_hidden_name=st.session_state.activation_hidden,
        activation_output_name=st.session_state.activation_output
    )
    st.plotly_chart(mlp_fig, use_container_width=True)

with col_activation_plot:
    st.subheader(f"Hidden: {st.session_state.activation_hidden} | Output: {st.session_state.activation_output}")
    # Plot both, or make selectable? For now, just one.
    st.markdown("**Hidden Layer Activation Function**")
    plot_activation_function_graph(st.session_state.activation_hidden)
    st.markdown("**Output Layer Activation Function**")
    plot_activation_function_graph(st.session_state.activation_output)


st.markdown("---")
st.header("🚀 Test Your MLP")

# Ensure user_inputs_values dict matches num_inputs
if len(st.session_state.user_inputs_values) != st.session_state.num_inputs:
    st.session_state.user_inputs_values = {f"x{i+1}": 0.0 for i in range(st.session_state.num_inputs)}
    reset_simulation_state() # Reset if input structure changed

input_cols = st.columns(st.session_state.num_inputs)
for i in range(st.session_state.num_inputs):
    input_key = f"x{i+1}"
    st.session_state.user_inputs_values[input_key] = input_cols[i].number_input(
        f"Input x{i+1}:", value=float(st.session_state.user_inputs_values.get(input_key, 0.0)),
        step=0.1, key=f"user_input_{i}", format="%.2f"
    )

current_input_list = [st.session_state.user_inputs_values[f"x{i+1}"] for i in range(st.session_state.num_inputs)]

if st.button("▶️ Run MLP with Current Inputs", key="run_button", type="primary"):
    st.session_state.run_simulation = True
    output_array, all_layer_activations = mlp.predict(np.array(current_input_list))
    
    if output_array is not None:
        st.session_state.current_mlp_output = output_array
        st.session_state.layer_activations_cache = all_layer_activations
        st.rerun() # Re-run to update the diagram with activations
    else:
        st.error("Could not compute MLP output.")
        reset_simulation_state()
        st.rerun()

# Enhanced Output Display Tile
if st.session_state.run_simulation and st.session_state.current_mlp_output is not None:
    st.markdown("---")
    
    # Create a tile-style output display
    output_tile = st.container()
    with output_tile:
        st.subheader("📊 MLP Output Results")
        
        # Create columns for better layout
        col_input, col_output = st.columns(2)
        
        with col_input:
            st.markdown("**Input Values**")
            input_values_str = ", ".join([f"{val:.2f}" for val in current_input_list])
            st.markdown(f"`[{input_values_str}]`")
            
        with col_output:
            st.markdown("**MLP Output**")
            output_array = st.session_state.current_mlp_output
            
            # For single output
            if output_array.shape[1] == 1:
                output_value = output_array[0,0]
                # Color code based on activation function
                if st.session_state.activation_output in ["Sigmoid", "Step"]:
                    color = "green" if output_value >= 0.5 else "red"
                    interpretation = "Active (≥0.5)" if output_value >= 0.5 else "Inactive (<0.5)"
                else:
                    color = "blue"
                    interpretation = ""
                
                st.markdown(f"""
                <div style="
                    background: rgba(240, 242, 246, 0.5);
                    border-radius: 10px;
                    padding: 15px;
                    border-left: 5px solid {color};
                    margin-bottom: 10px;
                ">
                    <h3 style="color: {color}; margin-top: 0;">{output_value:.4f}</h3>
                    <p>{interpretation}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # For multiple outputs
            else:
                for i in range(output_array.shape[1]):
                    output_value = output_array[0,i]
                    st.markdown(f"""
                    <div style="
                        background: rgba(240, 242, 246, 0.5);
                        border-radius: 10px;
                        padding: 10px;
                        border-left: 5px solid blue;
                        margin-bottom: 10px;
                    ">
                        <b>Output {i+1}:</b> {output_value:.4f}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Add activation function info
        st.caption(f"Output activation function: {st.session_state.activation_output}")
        
        # For classification outputs, add interpretation
        if st.session_state.activation_output in ["Sigmoid", "Step"] and output_array.shape[1] == 1:
            threshold = 0.5
            prediction = 1 if output_array[0,0] >= threshold else 0
            st.info(f"Interpretation (threshold={threshold}): Predicted class = {prediction}")

# Logic Gate Testing Section
if selected_gate != "Custom":
    st.markdown("---")
    st.header(f"🔬 Logic Gate Analysis: {selected_gate.upper()}")
    
    gate_info = gate_data[selected_gate]
    gate_inputs_np = gate_info["inputs"]
    gate_targets_np = gate_info["outputs"]

    st.markdown(f"**Truth Table & MLP Predictions for {selected_gate.upper()}:**")
    
    table_data_list = []
    for i_idx, inp_row in enumerate(gate_inputs_np):
        mlp_out_arr, _ = mlp.predict(inp_row)
        
        # Format MLP output for display (handle multiple output neurons if any)
        mlp_out_str = ""
        if mlp_out_arr is not None:
            if mlp_out_arr.shape[1] > 1:
                 mlp_out_str = ", ".join([f"{val:.2f}" for val in mlp_out_arr[0]])
            else:
                 mlp_out_str = f"{mlp_out_arr[0,0]:.2f}"
        else:
            mlp_out_str = "Error"

        table_data_list.append({
            "Inputs": str(list(inp_row)), 
            "Target Output": str(gate_targets_np[i_idx]), # Assuming single target output for gates
            "MLP Output": mlp_out_str
        })
        
    st.dataframe(table_data_list, use_container_width=True)

    if "note" in gate_info:
        st.info(f"Note: {gate_info['note']}")

    if gate_info["num_inputs"] == 2 and mlp.layer_config[-1] == 1: # Plot boundary for 2-input, 1-output MLP
        st.subheader("Decision Boundary Plot")
        plot_decision_boundary_mlp(gate_inputs_np, gate_targets_np, mlp, st.session_state.activation_output)
else:
    if st.session_state.num_inputs == 2 and st.session_state.num_outputs == 1:
        st.markdown("---")
        st.subheader("📊 Custom 2D Input Space (Optional)")
        st.markdown("For a custom 2-input, 1-output MLP, the decision boundary plot can be generated if you provide sample data points and targets. This feature is primarily demonstrated with logic gates for now.")

st.sidebar.markdown("---")
st.sidebar.info("Weights & biases are auto-initialized based on architecture. Full MLP training is not part of this visualizer.")
st.sidebar.markdown("Created with Streamlit, Plotly & NumPy.")