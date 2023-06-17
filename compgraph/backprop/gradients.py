class Gradients:
    """
    Utility class for computing gradients in a computational graph.
    """

    @staticmethod
    def compute_gradients(loss: 'Node'):
        """
        Computes gradients of the loss with respect to all nodes in the computational graph.

        Parameters:
            loss: The loss node in the graph.

        Returns:
            dict: A dictionary mapping nodes to their corresponding gradients.

        Example:
            import compgraph as cg
            from compgraph.backprop import Gradient


            p = cg.Constant([[0.7, 0.3], [0.9, 0.1]])
            y = cg.Constant([[1, 0], [0, 1]])
            cross_entropy = -cg.reduce_sum(cg.reduce_sum(y * cg.log(p), axis=1))
            gradients = Gradients.compute_gradients(loss)

            print(gradients)
            # Outputs: (pretty printed)
            {
                -(sigmoid(p)): 1,

                sigmoid(p): -1,

                p: [[-0.19661193 -0.25      ]
                    [-0.25       -0.19661193]],
            }
        """
        gradients = {}
        gradients[loss] = 1

        lolol = list(reversed(loss._graph.get_topological(loss)[:-1]))

        for node in lolol:
            gradients[node] = 0

            for consumer in node.consumers:
                try:
                    gradient_wrt_consumer = consumer.gradient(
                        gradients[consumer])

                except KeyError:
                    print(f'consumer={consumer}', lolol.index(consumer))
                    raise

                if len(consumer.input_nodes) == 1:
                    gradients[node] += gradient_wrt_consumer
                else:
                    gradients[node] += gradient_wrt_consumer[consumer.input_nodes.index(
                        node)]

        return gradients
