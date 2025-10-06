#!/usr/bin/env python3

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from math import ceil
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

plt.style.use("./computermodern.mplstyle")
matplotlib.rcParams.update({
    "xtick.bottom": False, "xtick.labelbottom": False,
    "ytick.left":   False, "ytick.labelleft":   False
})

"""
Model architecture:

   n_inputs=2  n_hl_1=4  n_hl_2=4

    x0 .          .         .

                  .         .
                                    y0 .
                  .         .

    x1 .          .         .
"""

#  ---training examples---
X_train = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
)
n_examples = X_train.shape[0]

Y_train = np.array([0, 1, 1, 0])

assert X_train.shape[0] == Y_train.shape[0], "Training example mismatch."
# ---


class MLP:
    def __init__(self, n_inputs=2, hl=[4, 4], n_outputs=1):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hl = len(hl)

        self.layers = [n_inputs, *hl, n_outputs]
        self.W = []
        self.B = []
        for i in range(1, 2 + self.n_hl):
            self.W.append(np.random.random((self.layers[i], self.layers[i - 1])))
            self.B.append(np.zeros(shape=(self.layers[i], 1)))
        #
        # example for 2 hidden layers
        # W0 = np.random.random((hl[0], n_inputs))
        # W1 = np.random.random((hl[1], hl[0]))
        # W2 = np.random.random((n_outputs, hl[1]))
        # b0 = np.zeros(shape=(hl[0], 1))
        # b1 = np.zeros(shape=(hl[1], 1))
        # b2 = np.zeros(shape=(n_outputs, 1))
        self.lr = 0.1

    def forward(self, X):
        Z_compute_graph = []

        activations = [X]  # X is the input layer, shape=(n_inputs, 1)
        for i in range(self.n_hl + 1):
            Z_compute_graph.append(model.W[i] @ activations[-1] + model.B[i])
            activations.append(activation(Z_compute_graph[-1]))

            # # ===***=== Forward pass ===***===
            # # example for 2 hidden layers
            # Z1 = model.W[0] @ X + model.B[0]  # (n_hl_1, n_inputs) x (n_inputs, 1) + (n_hl_1, 1)
            # A1 = activation(Z1)  # (n_hl_1, 1)
            # Z2 = model.W[1] @ A1 + model.B[1]  # (n_hl_2, n_hl_1) x (n_hl_1, 1) + (n_hl_2, 1)
            # A2 = activation(Z2)  # (n_hl_2, 1)
            # Z3 = model.W[2] @ A2 + model.B[2]  # (n_outputs, n_hl_2) x (n_hl_2, 1) + (n_outputs, 1)
            # A3 = activation(Z3)  # (n_outputs, 1)
            # #            ===***===

        return Z_compute_graph, activations

    def dump_matrices(self, X, Y, figname, show=False):
        """ For each layer, output a picture of the weights and biases. """

        fig = plt.figure(figsize=(16, 9), dpi=160)
        n_matrices = self.n_hl + 1

        # I want 3 matrices per row
        nrows = ceil(n_matrices / 3)
        ncols = 3
        weight_heights = [4] * nrows
        bias_heights = [1] * nrows
        gs = GridSpec(
                nrows=nrows * 2 + 2,  # 2xnrows for weights & biases, 2 extra for input and output layers
                ncols=ncols + 1,  # one extra col for a colorbar
                height_ratios=[1, *weight_heights, *bias_heights, 1],
                width_ratios=[3, 3, 3, 1], # [4, *([3] * 3), *([3] * 3), 4],
        )
        ax_cbar = fig.add_subplot(gs[1:-1, -1])
        norm = Normalize(vmin=-1, vmax=1)
        sm = plt.cm.ScalarMappable(norm=norm, cmap="twilight_shifted")
        sm.set_array([])  # harmless placeholder for older Matplotlib
        cb = fig.colorbar(sm, ax_cbar, orientation="vertical")
        cb.set_label("Intensity")

        ax_input = fig.add_subplot(gs[0, :-1])
        ax_input.set_title("Input Layer")
        ax_input.matshow(X.reshape(self.n_inputs, 1), cmap="twilight_shifted")

        ax_output = fig.add_subplot(gs[-1, :-1])
        ax_output.set_title("Output Layer")
        ax_output.matshow(Y.reshape(self.n_outputs, 1), cmap="twilight_shifted")

        for i in range(n_matrices):
            ax = fig.add_subplot(gs[i // 3 + 1, i % 3])
            ax.matshow(model.W[i], cmap="twilight_shifted")

            if i == 1:
                ax.set_title("Weights")

            ax = fig.add_subplot(gs[nrows + i // 3 + 1, i % 3])
            ax.matshow(model.B[i].T, cmap="twilight_shifted")
            if i == 1:
                ax.set_title("Biases")

        # fig.tight_layout()
        plt.savefig(figname)
        if show:
            plt.show()


    # -- GPT magic ---
    def dump_state(self):
        """
        Returns a plain dict you can serialize/log each step:
        - weights, biases, Zs, As (deep copies).
        """
        Zs, As = self.forward(
            self._last_X_for_dump
            if hasattr(self, "_last_X_for_dump")
            else np.zeros((self.n_inputs, 1))
        )
        return {
            "layers": list(self.layers),
            "W": [w.copy() for w in self.W],
            "B": [b.copy() for b in self.B],
            "Z": [z.copy() for z in Zs],
            "A": [a.copy() for a in As],
        }

    # GRAPH OUTPUT
    def graph_out(
        self,
        X,
        filename=None,
        title="MLP Activations",
        figsize=(10, 6),
        node_cmap="twilight_shifted",
        edge_cmap="coolwarm",
        node_size_base=600,
        edge_max_width=4.0,
        edge_alpha=0.7,
        show=True,
        annotate_values=True,
        zero_center_edges=True,
    ):
        """
        Render a 3b1b-style network diagram.
        - Node color intensity = activation magnitude in that layer.
        - Edge color = sign of weight (red negative, blue positive via 'coolwarm'); width ∝ |weight|.
        - Each layer is labeled; node text shows activation (if annotate_values=True).

        Args:
            X: np.ndarray, shape (n_inputs, 1)
            filename: if provided, saves the figure to this path
            title: figure title
            figsize: matplotlib figsize
            node_cmap, edge_cmap: colormaps for nodes/edges
            node_size_base: base scatter size for nodes
            edge_max_width: max linewidth for the thickest edge
            edge_alpha: edge transparency
            show: whether to plt.show()
            annotate_values: annotate activations under/inside nodes
            zero_center_edges: normalize edges around 0 (so 0→mid color)
        Returns:
            fig, ax (matplotlib objects)
        """
        # Run a forward pass to get activations
        Zs, As = self.forward(X)
        self._last_X_for_dump = X  # so dump_state() can reproduce

        # Layout: x = layer index; y = evenly spaced neuron positions (top→bottom)
        layer_sizes = self.layers
        n_layers = len(layer_sizes)

        # Compute positions for all nodes: [(x, y)] per layer/neuron
        x_spacing = 1.0
        y_spacing = 1.0
        positions = []  # positions[l][j] = (x, y)

        def layer_y_positions(n):
            # center nodes vertically; top at positive y
            if n == 1:
                return np.array([0.0])
            return np.linspace((n - 1) * 0.5, -(n - 1) * 0.5, n)

        for l, n_l in enumerate(layer_sizes):
            x = l * x_spacing
            ys = layer_y_positions(n_l)
            positions.append([(x, y_spacing * y) for y in ys])

        # Normalize node colors per layer (3b1b "intensity" feel)
        node_norms = []
        for l in range(n_layers):
            A_l = As[l] if l < len(As) else None  # A[0]=input; A[L]=output
            if A_l is None:
                node_norms.append(np.zeros((layer_sizes[l], 1)))
                continue
            # Scale activations for color: per-layer min/max → [0,1]
            v = A_l.reshape(-1, 1)
            vmin, vmax = float(np.min(v)), float(np.max(v))
            if np.isclose(vmax, vmin):
                # flat layer: put at 0.5 intensity
                normed = np.ones_like(v) * 0.5
            else:
                normed = (v - vmin) / (vmax - vmin)
            node_norms.append(normed)

        # Edge normalization (for thickness + color)
        all_abs_w = np.concatenate([np.abs(Wi).ravel() for Wi in self.W])
        w_abs_max = np.max(all_abs_w) if all_abs_w.size > 0 else 1.0
        if zero_center_edges:
            all_w = np.concatenate([Wi.ravel() for Wi in self.W])
            w_abs_max = max(np.max(np.abs(all_w)), 1e-8)  # symmetric around 0

        node_cmap = matplotlib.colormaps[node_cmap]
        edge_cmap = matplotlib.colormaps[edge_cmap]

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)

        # Draw edges (between layer l and l+1)
        for l in range(n_layers - 1):
            W_l = self.W[l]  # shape: (n_{l+1}, n_l)
            for j in range(W_l.shape[0]):  # j: neuron index in layer l+1
                x2, y2 = positions[l + 1][j]
                for k in range(W_l.shape[1]):  # k: neuron index in layer l
                    x1, y1 = positions[l][k]
                    w = W_l[j, k]
                    if zero_center_edges:
                        edge_t = 0.5 + 0.5 * (
                            w / (w_abs_max + 1e-8)
                        )  # map [-max,max] → [0,1]
                    else:
                        edge_t = np.abs(w) / (w_abs_max + 1e-8)  # [0,1]
                    color = edge_cmap(edge_t)
                    width = edge_max_width * (np.abs(w) / (w_abs_max + 1e-8))
                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        linewidth=width,
                        color=color,
                        alpha=edge_alpha,
                    )

        # Draw nodes
        for l in range(n_layers):
            xs = [p[0] for p in positions[l]]
            ys = [p[1] for p in positions[l]]
            colors = node_cmap(node_norms[l].ravel())
            sizes = np.full(len(xs), node_size_base)

            ax.scatter(
                xs, ys, s=sizes, c=colors, edgecolors="k", linewidths=0.8, zorder=3
            )

            # Layer label above layer
            layer_name = (
                "Input" if l == 0 else "Output" if l == n_layers - 1 else f"Hidden {l}"
            )
            ax.text(
                xs[0],
                max(ys) + 0.6,
                f"{layer_name}\nA^{[{l}]}",  # superscript-ish label
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

            # Optional: annotate activations
            if annotate_values:
                A_l = As[l]
                for idx, (x, y) in enumerate(positions[l]):
                    val = float(A_l[idx]) if l < len(As) else 0.0
                    ax.text(
                        x,
                        y,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white",
                    )

        ax.axis("off")
        ax.set_aspect("equal", adjustable="datalim")
        plt.tight_layout()

        if filename is not None:
            fig.savefig(filename, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax

    # -- ---


# ---

model = MLP()


def activation(arr):
    return 1 / (1 + np.exp(-1 * arr))


def activation_prime(arr):
    # sigmoid' = exp(-arr)/(1+exp(-arr))**2
    # or more efficiently,
    x = activation(arr)
    return x * (1 - x)


m = n_examples
for t in range(n_examples):
    print(
        f"==== Training Step {t}\n{model.W[0]=},\n{model.W[1]=},\n{model.W[2]=},\n{model.B[0]=},\n{model.B[1]=},\n{model.B[2]=},\n{model.lr=}\n"
    )
    print("=== *** ===")
    X = X_train[t].reshape(model.n_inputs, 1)
    Y = Y_train[t].reshape(model.n_outputs, 1)

    # ===***=== Forward pass ===***===
    Z_compute_graph, activations = model.forward(X)
    Z1, Z2, Z3 = Z_compute_graph
    _, A1, A2, A3 = activations
    #             ===***===

    # ===***=== Loss ===***===
    C = (1 / m) * np.sum(np.square(A3 - Y))
    print(f"Training step #{t}\nAvg MSE loss={C}\n")
    #        ===***===

    # ===***=== Backprop ===***===
    """
    delta_3 = d(C)/d(Z3) = [d(C)/d(A3)] . sigmoid'(Z3)  # special for output layer
    """

    delta_3 = (A3 - Y) * activation_prime(Z3)
    dC_dW3 = (1 / m) * (delta_3 @ A2.T)
    dC_db3 = (1 / m) * np.sum(delta_3, axis=1, keepdims=True)

    dC_dA2 = model.W[2].T @ delta_3
    delta_2 = dC_dA2 * activation_prime(Z2)
    dC_dW2 = (1 / m) * (delta_2 @ A1.T)
    dC_db2 = (1 / m) * np.sum(delta_2, axis=1, keepdims=True)

    dC_dA1 = model.W[1].T @ delta_2
    delta_1 = dC_dA1 * activation_prime(Z1)
    print(f"xxxx {delta_1.shape=}{X.shape=}")
    dC_dW1 = (1 / m) * (delta_1 @ X.T)
    dC_db1 = (1 / m) * np.sum(delta_1, axis=1, keepdims=True)
    #         ===***===

    # ===***=== Gradient descent update ===***===
    model.W[2] -= model.lr * dC_dW3
    model.W[1] -= model.lr * dC_dW2
    print(f"xxxx {(model.lr * dC_dW2).shape=}")
    print(f"xxxx {(model.lr * dC_dW1).shape=}")
    model.W[0] -= model.lr * dC_dW1

    model.B[2] -= model.lr * dC_db3
    model.B[1] -= model.lr * dC_db2
    model.B[0] -= model.lr * dC_db1
    #                   ===***===

    # Make an image of the current model state
    model.graph_out(X, filename=f"step_{t}", show=False)
    model.dump_matrices(X, Y, figname=f"mat_step_{t}")

test = np.array([0, 0])
_, a = model.forward(test)
print("\n\n\n")
print(f"Model prediction for {test=}: {a[-1][-1]=}")
