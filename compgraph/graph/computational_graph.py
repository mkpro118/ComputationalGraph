import numpy as np
from itertools import chain

from compgraph.backprop.optimizers import OptimizerABC


class ComputationalGraph:
    """
    Encapsulates a computational graph.

    Attributes:
        constants (list): List of constant nodes in the graph.
        operations (list): List of operation nodes in the graph.
        placeholders (list): List of placeholder nodes in the graph.
        variables (list): List of variable nodes in the graph.

    Example:
        import compgraph as cg

        graph = cg.ComputationalGraph().as_default()
    """

    _default_graph = None

    def __init__(self):
        """
        Initializes a new instance of the ComputationalGraph class.
        """
        self.constants: list = list()
        self.operations: list = list()
        self.placeholders: list = list()
        self.variables: list = list()

    def as_default(self):
        """
        Set this graph as the default graph for the current session.
        """
        ComputationalGraph._default_graph = self

        return self

    def get_topological(self,
                        from_node: 'compgraph.graph.nodes.Operation' = None,
                        *,
                        disconnected: bool = False) -> list:
        """
        Returns a topologically sorted list of nodes starting from the given node.

        Parameters:
            from_node (compgraph.graph.nodes.Operation): The starting node (optional).
            disconnected (bool): Flag indicating whether to include all nodes in the graph (optional).

        Returns:
            list: The topologically sorted list of nodes.

        Example:
            import compgraph as cg

            graph = cg.ComputationalGraph()

            x = cg.Variable(5, name='x')
            y = cg.Variable(2, name='y')

            z = 2 * x + y

            topological = graph.get_topological(z)

            print(topological)
            # Outputs: [2, x, 2 * x, y, (2 * x) + y]
        """
        self.from_node = from_node

        topological: list = []

        def traverse(node):
            if hasattr(node, 'input_nodes'):
                for input_node in node.input_nodes:
                    traverse(input_node)
            topological.append(node)

        if from_node is not None:
            traverse(from_node)

        if disconnected:
            topological = [i for i in chain(
                self.constants,
                self.operations,
                self.placeholders,
                self.variables
            )] + topological

        return topological

    def visualize(self) -> None:
        """
        Visualizes the computational graph using a matplotlib plot.

        Raises:
            ValueError: If there are no operations in the graph.

        Example:
            import compgraph as cg
            graph = cg.ComputationalGraph().as_default()

            x = cg.Variable(5, name='x')
            y = cg.Variable(2, name='y')

            z = 2 * x + y

            graph.visualize()
            # Creates a matplotlib figure
        """
        from collections import deque, defaultdict
        from matplotlib import pyplot as plt

        if not self.operations:
            raise ValueError('Need at least 1 operation')

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        node = next(filter(lambda x: not isinstance(
            x, OptimizerABC), reversed(self.operations)))

        nodes = []
        # Queue to keep track of nodes and their levels
        queue = deque([(node, 0)])
        visited = set()  # Set to track visited nodes

        while queue:
            current_node, level = queue.popleft()

            if current_node in visited:
                continue

            visited.add(current_node)

            nodes.append([current_node, level])

            if hasattr(current_node, 'input_nodes'):
                for child in current_node.input_nodes:
                    if child not in visited:
                        queue.append((child, level + 1))

        self.max_level = max(nodes, key=lambda x: x[1])[1]

        for node in nodes:
            if not hasattr(node[0], 'input_nodes'):
                node[1] = self.max_level

        nodes.sort(key=lambda x: (x[1], x[0].name))

        levels = defaultdict(list)

        for i in range(self.max_level + 1):
            levels[i].extend(filter(lambda x: x[1] == i, nodes))

        ax.axis('off')

        positions = {}

        for level in levels.values():
            points = np.linspace(0, 5, num=len(levels)).tolist()
            points.sort(key=lambda x: abs(2.5 - x))
            for i, node in enumerate(level):
                positions[node[0]] = (-points[i], node[1] - self.max_level,)

        for (node, _) in nodes:
            x, y = positions[node]
            ax.text(x, y, node.name, ha='center', va='center', bbox=dict(
                facecolor='white', edgecolor='black', boxstyle='square'))

            if hasattr(node, 'input_nodes'):
                for parent in node.input_nodes:
                    parent_x, parent_y = positions[parent]
                    ax.plot([parent_x, x], [parent_y, y], 'g-', lw=1)
