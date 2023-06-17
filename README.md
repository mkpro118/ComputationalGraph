# CompGraph

CompGraph is a Python library for creating and working with computational graphs. It provides a flexible framework for defining and evaluating mathematical expressions, enabling efficient computations with automatic differentiation.

# Features

- **Computational Graph**: Build and manipulate a computational graph representing mathematical expressions.
- **Automatic Differentiation**: Perform reverse mode automatic differentiation to compute gradients of expressions.
- **Element - wise Operations**: Support for element - wise operations such as addition, subtraction, multiplication, and division.
- **Matrix Operations**: Enable matrix operations including matrix multiplication.
- **Power and Negative Operations**: Handle exponentiation and negation of expressions.
- **Special Functions**: Supports log, sigmoid, softmax and reduce_sum(with support for more functions coming soon).
- **Optimizers**: Supports the Stochastic Gradient Descent(SGD) optimizer(with support for more optimizers coming soon).

# Installation

You can install CompGraph using `git clone`:

```bash

git clone https: // github.com / mkpro118 / ComputationalGraph.git

```

You can add ComputationalGraph to the `PYTHONPATH` environment variable to easily import it like any other python module

# Usage

Here's a simple example that demonstrates how to create a computational graph and evaluate an expression:

```python
import compgraph as cg

# Create input variables
x = cg.Variable(2.0)
y = cg.Variable(3.0)

# Define the computational graph
z = x * y + 5

print(z)  # Output: 11.0
```

[example.py](.\compgraph\example.py) has an example of a basic classifier built using a more complex computational graph and automatic differentiation. Here's a shorter version of the same

```python
import compgraph as cg
import numpy as np

# Create training input placeholder
X = cg.PlaceHolder(name='X')

# Create placeholder for the training classes
y = cg.PlaceHolder(name='y')

# Build a hidden layer
W_hidden = cg.Variable(np.random.randn(2, 2), name='W_h')
b_hidden = cg.Variable(np.random.randn(2), name='b_h')
p_hidden = cg.sigmoid(X @ W_hidden + b_hidden, name='p_h')

# Build the output layer
W_output = cg.Variable(np.random.randn(2, 2), name='W_o')
b_output = cg.Variable(np.random.randn(2), name='b_o')
p_output = cg.softmax(p_hidden @ W_output + b_output, name='p_o')

# Build cross-entropy loss
loss = -cg.reduce_sum(cg.reduce_sum(y * cg.log(p_output), axis=1))
loss.name = 'Cross Entropy Loss'

# Build minimization operation
minimization_op = cg.SGD(learning_rate=0.03).optimize(loss)

# Build placeholder inputs
feed_dict = {
    X: np.concatenate((blue_points, red_points)),
    y:
        [[1, 0]] * len(blue_points)
        + [[0, 1]] * len(red_points)

}

session = cg.Session()

iterations = 1000
losses = np.zeros((iterations,))

# Perform the gradient descent steps
for step in range(1, iterations + 1):
    losses[step - 1] = session.run(loss, feed_dict)
    if step % 100 == 0:
        print(f'Step: {step} Loss: {losses[step-1]}')
    session.run(minimization_op, feed_dict)
```
### The Data and the Classifier's decision boundary
![Classifier](.\classifier.png)

### The Computational Graph
![Computational Graph](.\computational_graph.png)

### Loss vs. Iterations
![Loss vs. Iterations](.\loss_v_iterations.png)


# Contributing
Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request on the GitHub repository.

# License
This project is licensed under the MIT License.
