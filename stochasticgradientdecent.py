"""
c.  Write a program to execute (non-stochastic) gradient descent for the linear
model on the dataset above. Each iteration of gradient descent should use the entire dataset to update
the weight parameters. As usual, let the cost function C for the entire dataset be the sum of the
hinge losses c for each training example. Run your program to convergence of the parameter values.
Compare the h learned here to the h learned in part (c).

"""
import numpy as np

# Dataset: (x, y) pairs
data = np.array([
    [-5, -1],
    [1, 1],
    [-2, -1],
    [2, -1],
    [4, 1],
    [7, 1]
])

# Initial weights
w0, w1 = 0, 0

# Learning rate
eta = 0.1

# Function to compute hinge loss gradient
def hinge_loss_gradients(x, y, w0, w1):
    y_hat = w0 + w1 * x
    if 1 - y * y_hat > 0:
        return -y, -y * x  # Gradient w.r.t w0 and w1
    else:
        return 0, 0

# Maximum number of iterations
max_iterations = 100

# For convergence check 
epsilon = 1e-6
prev_w0, prev_w1 = w0, w1

# Batch Gradient Descent
for iteration in range(max_iterations):
    grad_w0, grad_w1 = 0, 0
    
    # Sum gradients for all examples
    for x, y in data:
        dw0, dw1 = hinge_loss_gradients(x, y, w0, w1)
        grad_w0 += dw0
        grad_w1 += dw1
    
    # Update weights
    w0 -= eta * grad_w0
    w1 -= eta * grad_w1
    
    # Convergence check (if change in both weights is very small)
    if abs(w0 - prev_w0) < epsilon and abs(w1 - prev_w1) < epsilon:
        print(f"Converged after {iteration + 1} iterations.")
        break
    
    prev_w0, prev_w1 = w0, w1

# Output final weights
print(f"Final weights: w0 = {w0}, w1 = {w1}")
