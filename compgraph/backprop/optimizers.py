from compgraph.backprop.gradients import Gradients


class OptimizerABC:
    """
    Abstract base class for optimizers in the computational graph.
    """
    pass


class SGD:
    """
    Stochastic Gradient Descent optimizer for updating trainable variables in the computational graph.

    Parameters:
        learning_rate (float): The learning rate for the optimizer.

    Example:
        import compgraph as cg
        optimizer = cg.SGD(learning_rate=0.05)
        optimizer_instance = optimizer.optimize(loss)
    """

    def __init__(self, learning_rate: float = 5e-2):
        self.learning_rate = learning_rate

    def optimize(self, loss: 'compgraph.graph.nodes.Operation') -> 'Optimizer':
        from compgraph.graph.nodes import Operation, Variable
        learning_rate = self.learning_rate

        class Optimizer(Operation, OptimizerABC):
            def __call__(self):
                """
                Updates the trainable variables in the computational graph using Stochastic Gradient Descent.

                Example:
                    optimizer_instance()
                """
                gradients = Gradients.compute_gradients(loss)

                for node in filter(
                    lambda x: isinstance(x, Variable) and x.trainable,
                    gradients.keys()
                ):
                    node.value -= learning_rate * gradients[node]

        return Optimizer()
