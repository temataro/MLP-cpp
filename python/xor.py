#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from mlp import *

DBG = 0

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
    if DBG:
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

    # ===***=== Backprop & Step ===***===
    model.backward(m, Y, Z=Z_compute_graph, A=activations)

    # Make an image of the current model state
    model.graph_out(X, filename=f"step_{t}", show=False)
    model.dump_matrices(X, Y, figname=f"mat_step_{t}")

test = np.array([0, 0])
_, a = model.forward(test)
print(f"{a=}\n\n\n")

print(f"Model prediction for {test=}: {a[-1][-1]=}")
