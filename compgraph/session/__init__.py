import numpy as np

from compgraph.graph.nodes import Operation, Node, PlaceHolder, Variable


class Session:
    """
    An computational graph execution session.

    Example:
        import compgraph as cg
        x = cg.PlaceHolder(name='x')
        y = cg.PlaceHolder(name='x')
        operation = cg.add(x, y)

        inputs = {
            x: [1, 2],
            y: [3, 4],
        }

        session = cg.Session()
        result = session.run(operation, inputs)

        print(result)
        # Outputs: [4 6]
    """

    def run(self, operation: 'Operation', inputs: dict = None) -> np.ndarray:
        """
        Execute the given operation in the session.

        Parameters:
            operation (Operation): The operation to execute.
            inputs (dict): Dictionary of placeholder nodes and their corresponding values (optional).

        Returns:
            np.ndarray: The output of the operation.

        Raises:
            AssertionError: If the inputs are not of type dict.
            AssertionError: If the operation is not of type Node.
            KeyError: If a PlaceHolder node has no input value
            TypeError: If a node is not one of the expected types (Constant, Operation, PlaceHolder, Variable).
        """
        inputs = inputs or dict()

        assert isinstance(inputs, dict), (
            f"inputs must be dictionary, with keys being the placeholder nodes "
            f"and values being the corresponding placeholder values"
        )

        assert isinstance(operation, Node), (
            f"Only operations can be executed. found type {type(operation)}"
        )

        for node in operation._graph.get_topological(operation):
            if isinstance(node, PlaceHolder):
                try:
                    node.output = inputs[node]
                except KeyError:
                    raise KeyError(
                        f'PlaceHolder node {node} has no input value'
                    )
            elif isinstance(node, Variable):
                node.output = node.value
            elif isinstance(node, Operation):
                node.output = node(
                    *(input_node.output for input_node in node.input_nodes))
            else:
                raise TypeError(
                    f"Nodes must be one of the following types "
                    f"(Constant, Operation, PlaceHolder, Variable)"
                )

            if not isinstance(node.output, np.ndarray):
                node.output = np.array(node.output)

            node.value = node.output
        return operation.output
