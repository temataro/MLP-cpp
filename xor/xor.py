#!/usr/bin/env python3

import numpy as np

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
        # W0 = np.random.random((hl[0], n_inputs))
        # W1 = np.random.random((hl[1], hl[0]))
        # W2 = np.random.random((n_outputs, hl[1]))
        # b0 = np.zeros(shape=(hl[0], 1))
        # b1 = np.zeros(shape=(hl[1], 1))
        # b2 = np.zeros(shape=(n_outputs, 1))
        self.lr = 0.1


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

    # TODO: make this into a function on a model M
    # `M.forward(X)`
    # ===***=== Forward pass ===***===
    Z1 = model.W[0] @ X + model.B[0]  # (n_hl_1, n_inputs) x (n_inputs, 1) + (n_hl_1, 1)
    A1 = activation(Z1)  # (n_hl_1, 1)
    Z2 = model.W[1] @ A1 + model.B[1]  # (n_hl_2, n_hl_1) x (n_hl_1, 1) + (n_hl_2, 1)
    A2 = activation(Z2)  # (n_hl_2, 1)
    Z3 = model.W[2] @ A2 + model.B[2]  # (n_outputs, n_hl_2) x (n_hl_2, 1) + (n_outputs, 1)
    A3 = activation(Z3)  # (n_outputs, 1)
    #            ===***===

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
