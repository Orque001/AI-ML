import math
from math import sin, pi
from random import random
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return sin(pi * x)


def generate_training_examples(n=2):
    xs = [random() * 2 - 1 for _ in range(n)]
    return [(x, f(x)) for x in xs]


def fit_without_reg(examples):
    """Computes values of w0 and w1 that minimize the sum-of-squared-errors cost function

    Args:
    - examples: a list of two (x, y) tuples, where x is the feature and y is the label
    """
    w0 = 0
    w1 = 0
    ## BEGIN YOUR CODE ##
    (x1, y1), (x2, y2) = examples
    w1 = (y2 - y1) / (x2 - x1)
    w0 = y1 - w1 * x1
    return w0, w1
    ## END YOUR CODE ##
    return w0, w1


def fit_with_reg(examples, λ, step_size = 0.05, updates=1000):
    """Computes values of w0 and w1 that minimize the regularized sum-of-squared-errors cost function

    Args:
    - examples: a list of two (x, y) tuples, where x is the feature and y is the label
    - lambda_hp: a float representing the value of the lambda hyperparameter; a larger value means more regularization
    """
    (x1, y1), (x2, y2) = examples
    w0 = 0
    w1 = 0
    ## BEGIN YOUR CODE ##
    for _ in range(updates):
        dw0 = -2 * (y1 - w0 - w1 * x1) - 2 * (y2 - w0 - w1 * x2) + 2 * λ * w0
        dw1 = -2 * (y1 - w0 - w1 * x1) * x1 - 2 * (y2 - w0 - w1 * x2) * x2 + 2 * λ * w1
        w0 -= step_size * dw0
        w1 -= step_size * dw1
    ## END YOUR CODE ##
    return (w0, w1)


def test_error(w0, w1):
    n = 100
    xs = [i/n for i in range(-n, n + 1)]
    return sum((w0 + w1 * x - f(x)) ** 2 for x in xs) / len(xs)

def run_experiment(trials=1000, step_size=0.05, λ=1, updates=1000):
    errors_without_reg = []
    errors_with_reg = []
    lines_without_reg = []
    lines_with_reg = []

    for _ in range(trials):
        examples = generate_training_examples()

        w0_without_reg, w1_without_reg = fit_without_reg(examples)
        lines_without_reg.append((w0_without_reg, w1_without_reg))

        w0_with_reg, w1_with_reg = fit_with_reg(examples, λ, step_size, updates)
        lines_with_reg.append((w0_with_reg, w1_with_reg))

        error_without_reg = test_error(w0_without_reg, w1_without_reg)
        error_with_reg = test_error(w0_with_reg, w1_with_reg)

        errors_without_reg.append(error_without_reg)
        errors_with_reg.append(error_with_reg)

    avg_error_without_reg = sum(errors_without_reg) / len(errors_without_reg)
    avg_error_with_reg = sum(errors_with_reg) / len(errors_with_reg)

    return avg_error_without_reg, avg_error_with_reg, lines_without_reg, lines_with_reg


def plot_graph(lines_without_reg, lines_with_reg):
    x_vals = np.linspace(-1, 1, 200)
    f_vals = [f(x) for x in x_vals]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x_vals, f_vals, 'b-', linewidth=2)
    for w0, w1 in lines_without_reg:
        plt.plot(x_vals, w0 + w1 * x_vals, 'k-', linewidth=0.3, alpha=0.1)
    plt.title('without regularization')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(1, 2, 2)
    plt.plot(x_vals, f_vals, 'b-', linewidth=2)
    for w0, w1 in lines_with_reg:
        plt.plot(x_vals, w0 + w1 * x_vals, 'k-', linewidth=0.3, alpha=0.1)
    plt.title('with regularization')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()

# Run the experiment
if __name__ == "__main__":
    avg_error_without_reg, avg_error_with_reg, lines_without_reg, lines_with_reg = run_experiment()

    print(f"Average test error over 1000 trials without regularization: {avg_error_without_reg}")
    print(f"Average test error over 1000 trials with regularization: {avg_error_with_reg}")

    plot_graph(lines_without_reg, lines_with_reg)
