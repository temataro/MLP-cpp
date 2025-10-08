
# ===***=== Forward pass ===***===
# example for 2 hidden layers
Z1 = model.W[0] @ X + model.B[0]  # (n_hl_1, n_inputs) x (n_inputs, 1) + (n_hl_1, 1)
A1 = activation(Z1)  # (n_hl_1, 1)
Z2 = model.W[1] @ A1 + model.B[1]  # (n_hl_2, n_hl_1) x (n_hl_1, 1) + (n_hl_2, 1)
A2 = activation(Z2)  # (n_hl_2, 1)
Z3 = model.W[2] @ A2 + model.B[2]  # (n_outputs, n_hl_2) x (n_hl_2, 1) + (n_outputs, 1)
A3 = activation(Z3)  # (n_outputs, 1)
#            ===***===


# # ===***=== Backprop ===***===
# Example for a 2, 4, 4, 1 network
"""
delta_1.shape=(4, 1)
dC_db1.shape=(4, 1)
dC_dA1.shape=(4, 1)

dC_dA2.shape=(4, 1)
dC_dW1.shape=(4, 2)
delta_2.shape=(4, 1)
dC_db2.shape=(4, 1)
dC_dW2.shape=(4, 4)

delta_3.shape=(1, 1)
dC_dW3.shape=(1, 4)
(A3-Y).shape=(1, 1)
dC_db3.shape=(1, 1)
"""
# """
# delta_3 = d(C)/d(Z3) = [d(C)/d(A3)] . sigmoid'(Z3)  # special for output layer
# """
# # ===***=== Forward pass then backward pass ===***===
# # example for 2 hidden layers
# #            ===***===


# Z3 = model.W[2] @ A2 + model.B[2]  # (n_outputs, n_hl_2) x (n_hl_2, 1) + (n_outputs, 1)
# A3 = activation(Z3)  # (n_outputs, 1)
# ^  fwd step ===  backward step  v
# for completeness: dC_dA3 = A3 - Y
# delta_3 = dC_dA3 * activation_prime(Z3)  # (n_outputs, 1)
# dC_dW3 = (1 / m) * (delta_3 @ A2.T); dC_db3 = (1 / m) * np.sum(delta_3, axis=1, keepdims=True)
# ^ (n_outputs, 1) x (1, n_hl_2)        ^ (n_outputs, 1) --> sums along each row

# Z2 = model.W[1] @ A1 + model.B[1]  # (n_hl_2, n_hl_1) x (n_hl_1, 1) + (n_hl_2, 1)
# A2 = activation(Z2)  # (n_hl_2, 1)
# ^  fwd step ===  backward step  v
# dC_dA2 = model.W[2].T @ delta_3  # (n_hl_2, n_outputs) x (n_outputs, 1)
# delta_2 = dC_dA2 * activation_prime(Z2)  # (n_hl_2, 1) . (n_hl_2, 1)
# dC_dW2 = (1 / m) * (delta_2 @ A1.T)  # (n_hl_2, 1) x (1, n_hl_1)
# dC_db2 = (1 / m) * np.sum(delta_2, axis=1, keepdims=True)  # (n_outputs, 1)

# Z1 = model.W[0] @ X + model.B[0]  # (n_hl_1, n_inputs) x (n_inputs, 1) + (n_hl_1, 1)
# A1 = activation(Z1)  # (n_hl_1, 1)
# ^  fwd step ===  backward step  v

# dC_dA1 = model.W[1].T @ delta_2  # (n_hl_1, n_hl_2) x (n_hl_2, 1)
# delta_1 = dC_dA1 * activation_prime(Z1)  # (n_hl_1, 1) . (n_hl_1, 1)
# dC_dW1 = (1 / m) * (delta_1 @ X.T)  # (n_hl_1, 1) x (1, n_inputs)
# dC_db1 = (1 / m) * np.sum(delta_1, axis=1, keepdims=True)  # (n_hl_1, 1)

# Takeaways:
# At each step in the backprop process, there are five major variables

# to account for:
# (index notation: m= neuron in next layer; j= neuron in current layer;
# k=neuron in prev layer)
# done in prev computation:
# delta_m
# to compute now:
# delta_j
# dC_dAj
# dC_dWj
# dC_dbj
# # ===***=== Gradient descent update ===***===
# model.W[2] -= model.lr * dC_dW3
# model.W[1] -= model.lr * dC_dW2
# print(f"xxxx {(model.lr * dC_dW2).shape=}")
# print(f"xxxx {(model.lr * dC_dW1).shape=}")
# model.W[0] -= model.lr * dC_dW1

# model.B[2] -= model.lr * dC_db3
# model.B[1] -= model.lr * dC_db2
# model.B[0] -= model.lr * dC_db1
# #                   ===***===

