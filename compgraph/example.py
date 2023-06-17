from matplotlib import pyplot as plt

import numpy as np

import compgraph as cg

np.random.seed(118)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

# Create two clusters of red points centered at (0, 0) and (1, 1), respectively.
red_points = np.concatenate((
    0.2 * np.random.randn(25, 2) + np.array([[0, 0]] * 25),
    0.2 * np.random.randn(25, 2) + np.array([[1, 1]] * 25)
))

# Create two clusters of blue points centered at (0, 1) and (1, 0), respectively.
blue_points = np.concatenate((
    0.2 * np.random.randn(25, 2) + np.array([[0, 1]] * 25),
    0.2 * np.random.randn(25, 2) + np.array([[1, 0]] * 25)
))

ax1.scatter(red_points[:, 0], red_points[:, 1], color='red', marker='o')
ax1.scatter(blue_points[:, 0], blue_points[:, 1], color='blue', marker='o')

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

# Build minimization op
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

# Visualize classification boundary
xs = np.linspace(feed_dict[X][:, 0].min() - 0.1,
                 feed_dict[X][:, 0].max() + 0.1, num=100)
ys = np.linspace(feed_dict[X][:, 1].min() - 0.1,
                 feed_dict[X][:, 1].max() + 0.1, num=100)
pred_classes = []
for x in xs:
    for y in ys:
        pred_class = session.run(p_output,
                                 inputs={X: [[x, y]]})[0]
        pred_classes.append((x, y, pred_class.argmax()))
xs_p, ys_p = [], []
xs_n, ys_n = [], []
for x, y, c in pred_classes:
    if c == 0:
        xs_n.append(x)
        ys_n.append(y)
    else:
        xs_p.append(x)
        ys_p.append(y)
ax2.plot(xs_p, ys_p, 'r.', xs_n, ys_n, 'b.')

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
ax.set_xlabel('Iterations')
ax.set_ylabel('Cross Entropy Loss')
ax.plot(np.arange(1, iterations + 1), losses, color='lightblue')

cg.ComputationalGraph._default_graph.visualize()

plt.show()
