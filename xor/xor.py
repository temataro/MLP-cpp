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

n_inputs = 2
n_outputs = 1
n_hl_1 = 4
n_hl_2 = 4
W1 = np.random.random((n_hl_1, n_inputs))
W2 = np.random.random((n_hl_2, n_hl_1))
W3 = np.random.random((n_outputs, n_hl_2))
b1 = np.zeros(shape=(n_hl_1, 1))
b2 = np.zeros(shape=(n_hl_2, 1))
b3 = np.zeros(shape=(n_outputs, 1))
lr = 0.1
# ---


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
        f"==== Training Step {t}\n{W1=},\n{W2=},\n{W3=},\n{b1=},\n{b2=},\n{b3=},\n{lr=}\n"
    )
    print("=== *** ===")
    X = X_train[t].reshape(n_inputs, 1)
    Y = Y_train[t].reshape(n_outputs, 1)

    # TODO: make this into a function on a model M
    # `M.forward(X)`
    # ===***=== Forward pass ===***===
    Z1 = W1 @ X + b1  # (n_hl_1, n_inputs) x (n_inputs, 1) + (n_hl_1, 1)
    A1 = activation(Z1)  # (n_hl_1, 1)
    Z2 = W2 @ A1 + b2  # (n_hl_2, n_hl_1) x (n_hl_1, 1) + (n_hl_2, 1)
    A2 = activation(Z2)  # (n_hl_2, 1)
    Z3 = W3 @ A2 + b3  # (n_outputs, n_hl_2) x (n_hl_2, 1) + (n_outputs, 1)
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

    dC_dA2 = W3.T @ delta_3
    delta_2 = dC_dA2 * activation_prime(Z2)
    dC_dW2 = (1 / m) * (delta_2 @ A1.T)
    dC_db2 = (1 / m) * np.sum(delta_2, axis=1, keepdims=True)

    dC_dA1 = W2.T @ delta_2
    delta_1 = dC_dA1 * activation_prime(Z1)
    print(f"xxxx {delta_1.shape=}{X.shape=}")
    dC_dW1 = (1 / m) * (delta_1 @ X.T)
    dC_db1 = (1 / m) * np.sum(delta_1, axis=1, keepdims=True)
    #         ===***===

    # ===***=== Gradient descent update ===***===
    W3 -= lr * dC_dW3
    W2 -= lr * dC_dW2
    print(f"xxxx {(lr * dC_dW2).shape=}")
    print(f"xxxx {(lr * dC_dW1).shape=}")
    W1 -= lr * dC_dW1

    b3 -= lr * dC_db3
    b2 -= lr * dC_db2
    b1 -= lr * dC_db1
    #                   ===***===
