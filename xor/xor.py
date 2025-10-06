#!/usr/bin/env python3

import numpy as np

from mlp import *

plt.style.use("./computermodern.mplstyle")
matplotlib.rcParams.update(
    {
        "xtick.bottom": False,
        "xtick.labelbottom": False,
        "ytick.left": False,
        "ytick.labelleft": False,
    }
)

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

model = MLP()


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
