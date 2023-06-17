from typing import Iterable

import numpy as np

from compgraph.graph.nodes import Operation, Node
from compgraph.types import Numeric


class log(Operation):
    """
    Represents a Node that applies the logarithm function

    Parameters:
        input_node (Node): The input node.
        consumers (Iterable[Node]): Nodes that consume the output of this node. (optional)
        name (str): Name of the node (optional).

    Returns:
        Numeric: The result of applying the logarithm function to the input node's output.

    Example:
        import compgraph as cg
        import math

        x = cg.Variable(math.e)

        z = cg.log(x)

        session = cg.Session()

        result = session.run(z)

        print(result)
        # Outputs: 1.0
    """

    def __init__(self,
                 input_node: Node,
                 *,
                 consumers: Iterable[Node] = None,
                 name: str = None):
        super().__init__(
            [input_node],
            consumers=consumers,
            name=name or f'log({input_node.name})'
        )

    def __call__(self, input1: Numeric) -> Numeric:
        """
        Apply the logarithm function to the input.

        Parameters:
            input1 (Numeric): Input value. Usually the output of it's input nodes

        Returns:
            Numeric: Result of applying the logarithm function.
        """
        return np.log(input1)

    def gradient(self, grad: Numeric) -> np.ndarray:
        """
        Compute the gradient of the logarithm function.

        Parameters:
            grad (Numeric): Gradient value.

        Returns:
            np.ndarray: Gradient of the logarithm function.
        """
        return np.array(grad / self.input_nodes[0].output)


class sigmoid(Operation):
    """
    Represents a Node that applies the sigmoid function

    Parameters:
        input_node (Node): The input node.
        consumers (Iterable[Node]): Nodes that consume the output of this node.
        name (str): Name of the node (optional).

    Returns:
        Numeric: The result of applying the sigmoid function to the input node's output.

    Example:
        import compgraph as cg

        x = cg.Variable(0)
        z = cg.sigmoid(x)

        session = cg.Session()
        result = session.run(z)

        print(result)
        # Outputs: 0.5
    """

    def __init__(self,
                 input_node: Node,
                 *,
                 consumers: Iterable[Node] = None,
                 name: str = None):
        super().__init__(
            [input_node],
            consumers=consumers,
            name=name or f'sigmoid({input_node.name})'
        )

    def __call__(self, input1: Numeric) -> Numeric:
        """
        Apply the sigmoid function to the input.

        Parameters:
            input1 (Numeric): Input value.

        Returns:
            Numeric: Result of applying the sigmoid function.
        """
        return 1 / (1 + np.exp(-input1))

    def gradient(self, grad: Numeric) -> np.ndarray:
        """
        Compute the gradient of the sigmoid function.

        Parameters:
            grad (Numeric): Gradient value.

        Returns:
            np.ndarray: Gradient of the sigmoid function.
        """
        sigmoid = self.output

        return np.array(grad * sigmoid * (1 - sigmoid))


class softmax(Operation):
    """
    Represents a Node that applies the softmax function

    Parameters:
        input_node (Node): The input node.
        consumers (Iterable[Node]): Nodes that consume the output of this node.
        name (str): Name of the node (optional).

    Returns:
        Numeric: The result of applying the softmax function to the input node's output.

    Example:
        import compgraph as cg

        x = cg.Variable([[0, 0], [0, 0]])
        z = cg.softmax(x)

        session = cg.Session()
        result = session.run(z)

        print(result)
        # Outputs: [[0.5 0.5]
                    [0.5 0.5]]
    """

    def __init__(self,
                 input_node: Node,
                 *,
                 consumers: Iterable[Node] = None,
                 name: str = None):
        super().__init__(
            [input_node],
            consumers=consumers,
            name=name or f'softmax({input_node.name})'
        )

    def __call__(self, input1: Numeric) -> Numeric:
        """
        Apply the softmax function to the input.

        Parameters:
            input1 (Numeric): Input value.

        Returns:
            Numeric: Result of applying the softmax function.
        """
        return (_ := np.exp(input1)) / np.sum(_, axis=1)[:, None]

    def gradient(self, grad: Numeric):
        """
        Compute the gradient of the softmax function.

        Parameters:
            grad (Numeric): Gradient value.

        Returns:
            np.ndarray: Gradient of the softmax function.
        """
        softmax = self.output

        return np.array((grad - np.reshape(
            np.sum(grad * softmax, axis=1),
            [-1, 1]
        )) * softmax)


class reduce_sum(Operation):
    """
    Represents a Node that computes the sum of elements across dimensions of a tensor

    Parameters:
        input_node1 (Node): The input node.
        axis (Numeric): Axis along which to perform the sum (optional).
        consumers (Iterable[Node]): Nodes that consume the output of this node.
        name (str): Name of the node (optional).

    Returns:
        Numeric: The sum of elements along the specified axis.

    Example:
        import compgraph as cg

        x = cg.Variable([[1, 2], [3, 4]])
        z = cg.reduce_sum(x)

        session = cg.Session()
        result = session.run(z)

        print(result)
        # Outputs: 10

        z = cg.reduce_sum(x, axis=0)
        result = session.run(z)

        print(result)
        # Outputs: [4 6]

        z = cg.reduce_sum(x, axis=1)
        result = session.run(z)

        print(result)
        # Outputs: [3 7]
    """

    def __init__(self,
                 input_node1: Node,
                 axis: Numeric = None,
                 *,
                 consumers: Iterable[Node] = None,
                 name: str = None):
        super().__init__(
            [input_node1],
            consumers=consumers,
            name=name or f'reduce_sum({input_node1.name}, {axis=})'
        )
        self.axis = axis

    def __call__(self, input1: Numeric) -> Numeric:
        """
        Compute the sum of elements along the specified axis.

        Parameters:
            input1 (Numeric): Input tensor.

        Returns:
            Numeric: The sum of elements along the specified axis.
        """
        return np.sum(input1, self.axis)

    def gradient(self, grad: Numeric) -> np.ndarray:
        """
        Compute the gradient of the reduce_sum function.

        Parameters:
            grad (Numeric): Gradient value.

        Returns:
            np.ndarray: Gradient of the reduce_sum function.
        """
        A = self.input_nodes[0].output

        output_shape = np.array(A.shape)
        output_shape[self.axis] = 1

        tile_scaling = A.shape // output_shape
        grad = np.reshape(grad, output_shape)

        return np.tile(grad, tile_scaling)
