from typing import Iterable, Union

from functools import wraps
import numpy as np

from compgraph.cg_types import Numeric, NumericTypes
from compgraph.graph.computational_graph import ComputationalGraph


class _maybe_immediate:
    """
    Helper class that provides decorators for performing immediate operations on inputs.

    Default supported operations: 'add', 'subtract', 'multiply', 'divide', 'power', 'matmul', 'negative'

    """
    funcs = {
        'add': np.add,
        'subtract': np.subtract,
        'multiply': np.multiply,
        'divide': np.divide,
        'power': np.power,
        'matmul': np.matmul,
        'negative': np.negative,
    }

    symbols = {
        'add': '+',
        'subtract': '-',
        'multiply': '*',
        'divide': '/',
        'power': '**',
        'matmul': '@',
        'negative': '-',
    }

    def __init__(self, op):
        """
        Initializes a decorator for the specified operation.

        Args:
            op (str): The operation to perform.

        Returns:
            None
        """
        self.op = op

        def name(*ops):
            """
            Generates the name for the operation.

            Args:
                *ops: The operands of the operation.

            Returns:
                str: The name of the operation.

            """

            if self.op == 'negative':
                if isinstance(ops[0], np.ndarray) and ops[0].ndim > 0:
                    return f'-np.array(shape={ops[0].shape})'

                if isinstance(ops[0], Node):
                    return f'-({ops[0].name})'

                return f'-({ops[0]})'

            ops = map(
                lambda x: f'-np.array(shape={x.shape})' if isinstance(x, np.ndarray) and x.ndim > 0 else x, ops)

            ops = map(lambda x: f'({x.name})' if any(
                symbol in x.name for symbol in _maybe_immediate.symbols.values()) else x.name if isinstance(x, Node) else x, ops)

            ops = map(str, ops)

            return f' {_maybe_immediate.symbols[self.op]} '.join(ops)

        self.name = name

    def __call__(self, func):
        """
        Decorator function that performs the specified operation on the inputs.

        Args:
            func (callable): The function to decorate.

        Returns:
            callable: The decorated function.

        """
        @ wraps(func)
        def function(*ops):
            """
            Wrapper function that performs immediate operations on the inputs.

            Args:
                *ops: The operands of the operation.

            Returns:
                Node: The result of the operation.

            """
            new_node = None
            if all(isinstance(x, Constant) for x in ops):
                new_node = Constant(_maybe_immediate.funcs[self.op](
                    *map(lambda x: x.value, ops)), name=self.name(*ops))

            elif any((
                isinstance(ops[0], Variable) and isinstance(
                    ops[1], Constant),
                isinstance(ops[0], Constant) and isinstance(
                    ops[1], Variable)
            )):
                new_node = Variable(_maybe_immediate.funcs[self.op](
                    *map(lambda x: x.value, ops)), name=self.name(*ops))

            elif isinstance(ops[0], Variable) and isinstance(ops[1], NumericTypes):
                new_node = Variable(_maybe_immediate.funcs[self.op](
                    ops[0].value, ops[1]), name=self.name(*ops))

            elif isinstance(ops[0], Constant) and isinstance(ops[1], NumericTypes):
                new_node = Constant(_maybe_immediate.funcs[self.op](
                    ops[0].value, ops[1]), name=self.name(*ops))

            if new_node:
                for op in filter(lambda x: isinstance(x, Node), ops):
                    op.consumers.add(new_node)

                new_node.input_nodes = list(ops)

                return new_node

            return func(*ops)

        return function


class Node:
    """
    Base Class for all nodes in the computational graph.

    Attributes:
        n_anon_nodes (int): Counter for anonymous nodes.

    Methods:
        __init__(self, consumers=None, name=None)
        __str__(self)
        __repr__(self)
        __add__(self, other)
        __radd__(self, other)
        __sub__(self, other)
        __rsub__(self, other)
        __mul__(self, other)
        __rmul__(self, other)
        __truediv__(self, other)
        __rtruediv__(self, other)
        __pow__(self, other)
        __rpow__(self, other)
        __matmul__(self, other)
        __rmatmul__(self, other)
        __neg__(self)
    """

    n_anon_nodes: int = 0

    def __init__(self, *, consumers: Iterable['Node'] = None, name: str = None):
        """
        Initializes a Node in the computational graph.

        Parameters:
            consumers (Iterable[Node], optional): The nodes that consume the output of this node.
            name (str, optional): The name of the node.

        Example:
            from compgraph.graph.nodes import Node
            node = Node(name="node 1")
        """
        if name is None:
            self.name = f'{self.__class__.__name__}#{Node.n_anon_nodes}'
            Node.n_anon_nodes += 1
        else:
            assert isinstance(name, str), f"name must be of type 'str'"
            self.name = name

        self.consumers = set(consumers) if consumers else set()

        self._graph = ComputationalGraph._default_graph

    def __str__(self):
        """
        Returns the string representation of the node.

        Returns:
            str: The name of the node.
        """
        return f"cg.{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self):
        """
        Returns the string representation of the node.

        Returns:
            str: The name of the node.
        """
        return f"cg.{self.__class__.__name__}(name='{self.name}', #consumers = {len(self.consumers)})"

    @ _maybe_immediate('add')
    def __add__(self, other: Union['Node', Numeric]) -> 'add':
        """
        Adds the current node to another node or numeric value.

        Parameters:
            other (Union[Node, Numeric]): The other node or numeric value to add.

        Returns:
            add: The add operation node.

        Raises:
            TypeError: If the operation '+' is not supported between the types of self and other.

        Example:
            result = node1 + node2
        """
        if isinstance(other, Node):
            return add(self, other)

        if isinstance(other, NumericTypes):
            other = Constant(other)
            return self + other

        raise TypeError(
            f"'+' is not supported for types '{type(self)}' and '{type(other)}'"
        )

    def __radd__(self, other: Numeric) -> 'add':
        """
        Adds a numeric value to the current node.

        Parameters:
            other (Numeric): The numeric value to add.

        Returns:
            add: The add operation node.

        Raises:
            TypeError: If the operation '+' is not supported between the types of self and other.

        Example:
            result = 10 + node
        """
        if isinstance(other, NumericTypes):
            other = Constant(other)
            return other + self

        raise TypeError(
            f"'-' is not supported for types '{type(self)}' and '{type(other)}'"
        )

    @ _maybe_immediate('subtract')
    def __sub__(self, other: Union['Node', Numeric]) -> 'subtract':
        """
        Subtracts another node or numeric value from the current node.

        Parameters:
            other (Union[Node, Numeric]): The other node or numeric value to subtract.

        Returns:
            subtract: The subtract operation node.

        Raises:
            TypeError: If the operation '-' is not supported between the types of self and other.

        Example:
            result = node1 - node2
        """
        if isinstance(other, Node):
            return subtract(self, other)

        if isinstance(other, NumericTypes):
            other = Constant(other)
            return self - other

        raise TypeError(
            f"'-' is not supported for types '{type(self)}' and '{type(other)}'"
        )

    def __rsub__(self, other: Numeric) -> 'subtract':
        """
        Subtracts the current node from a numeric value.

        Parameters:
            other (Numeric): The numeric value to subtract from.

        Returns:
            subtract: The subtract operation node.

        Raises:
            TypeError: If the operation '-' is not supported between the types of self and other.

        Example:
            result = 10 - node
        """
        if isinstance(other, NumericTypes):
            other = Constant(other)
            return other - self

        raise TypeError(
            f"'-' is not supported for types '{type(self)}' and '{type(other)}'"
        )

    @ _maybe_immediate('multiply')
    def __mul__(self, other: Union['Node', Numeric]) -> 'multiply':
        """
        Multiplies the current node with another node or numeric value.

        Parameters:
            other (Union[Node, Numeric]): The other node or numeric value to multiply.

        Returns:
            multiply: The multiply operation node.

        Raises:
            TypeError: If the operation '*' is not supported between the types of self and other.

        Example:
            result = node1 * node2
        """
        if isinstance(other, Node):
            return multiply(self, other)

        if isinstance(other, NumericTypes):
            other = Constant(other)
            return self * other

        raise TypeError(
            f"'*' is not supported for types '{type(self)}' and '{type(other)}'"
        )

    def __rmul__(self, other: Numeric) -> 'multiply':
        """
        Multiplies a numeric value with the current node.

        Parameters:
            other (Numeric): The numeric value to multiply with.

        Returns:
            multiply: The multiply operation node.

        Raises:
            TypeError: If the operation '*' is not supported between the types of self and other.

        Example:
            result = 10 * node
        """
        if isinstance(other, NumericTypes):
            other = Constant(other)
            return other * self

        raise TypeError(
            f"'-' is not supported for types '{type(self)}' and '{type(other)}'"
        )

    @ _maybe_immediate('divide')
    def __truediv__(self, other: Union['Node', Numeric]) -> 'divide':
        """
        Divides the current node by another node or numeric value.

        Parameters:
            other (Union[Node, Numeric]): The other node or numeric value to divide by.

        Returns:
            divide: The divide operation node.

        Raises:
            TypeError: If the operation '/' is not supported between the types of self and other.

        Example:
            result = node1 / node2
        """
        if isinstance(other, Node):
            return divide(self, other)

        if isinstance(other, NumericTypes):
            other = Constant(other)
            return self / other

        raise TypeError(
            f"'/' is not supported for types '{type(self)}' and '{type(other)}'"
        )

    def __rtruediv__(self, other: Numeric) -> 'divide':
        """
        Divides a numeric value by the current node.

        Parameters:
            other (Numeric): The numeric value to divide by.

        Returns:
            divide: The divide operation node.

        Raises:
            TypeError: If the operation '/' is not supported between the types of self and other.

        Example:
            result = 10 / node
        """
        if isinstance(other, NumericTypes):
            other = Constant(other)
            return other / self

        raise TypeError(
            f"'-' is not supported for types '{type(self)}' and '{type(other)}'"
        )

    @ _maybe_immediate('power')
    def __pow__(self, other: Union['Node', Numeric]) -> 'power':
        """
        Raises the current node to the power of another node or numeric value.

        Parameters:
            other (Union[Node, Numeric]): The other node or numeric value to raise to the power of.

        Returns:
            power: The power operation node.

        Raises:
            TypeError: If the operation '**' is not supported between the types of self and other.

        Example:
            result = node1 ** node2
        """
        if isinstance(other, Node):
            return power(self, other)

        if isinstance(other, NumericTypes):
            other = Constant(other)
            return self ** other

        raise TypeError(
            f"'**' is not supported for types '{type(self)}' and '{type(other)}'"
        )

    def __rpow__(self, other: Numeric) -> 'power':
        """
        Raises a numeric value to the power of the current node.

        Parameters:
            other (Numeric): The numeric value to raise to the power of.

        Returns:
            power: The power operation node.

        Raises:
            TypeError: If the operation '**' is not supported between the types of self and other.

        Example:
            result = 10 ** node
        """
        if isinstance(other, NumericTypes):
            other = Constant(other)
            return other ** self

        raise TypeError(
            f"'**' is not supported for types '{type(self)}' and '{type(other)}'"
        )

    @ _maybe_immediate('matmul')
    def __matmul__(self, other: Union['Node', Numeric]) -> 'matmul':
        """
        Performs matrix multiplication between the current node and another node or numeric value.

        Parameters:
            other (Union[Node, Numeric]): The other node or numeric value for matrix multiplication.

        Returns:
            matmul: The matmul operation node.

        Raises:
            TypeError: If the operation '@' is not supported between the types of self and other.

        Example:
            result = node1 @ node2
        """
        if isinstance(other, Node):
            return matmul(self, other)

        if isinstance(other, NumericTypes):
            other = Constant(other)
            return self @ other

        raise TypeError(
            f"'@' is not supported for types '{type(self)}' and '{type(other)}'"
        )

    def __rmatmul__(self, other: Numeric) -> 'matmul':
        """
        Performs matrix multiplication of a numeric value with the current node.

        Parameters:
            other (Numeric): The numeric value for matrix multiplication.

        Returns:
            matmul: The matmul operation node.

        Raises:
            TypeError: If the operation '@' is not supported between the types of self and other.

        Example:
            result = np.array([[1, 0], [0, 1]]) @ node
        """
        if isinstance(other, NumericTypes):
            other = Constant(other)
            return other @ self

        raise TypeError(
            f"'@' is not supported for types '{type(self)}' and '{type(other)}'"
        )

    @ _maybe_immediate('negative')
    def __neg__(self) -> 'negative':
        """
        Negates the current node.

        Returns:
            negative: The negative operation node.

        Example:
            result = -node
        """
        return negative(self)


class PlaceHolder(Node):
    """
    Represents a placeholder node in the computational graph.

    Placeholders are used to provide input values to the graph during execution.

    Parameters:
        consumers (Iterable[Node]): Nodes that consume the output of this placeholder.
        name (str): Name of the placeholder.

    Attributes:
        n_anon_placeholders (int): Counter for anonymous placeholders.
        _graph (ComputationalGraph): The computational graph that the placeholder belongs to.

    Example:
        import compgraph as cg

        x = cg.PlaceHolder(name='x')  # Creating a named placeholder
        y = cg.PlaceHolder()  # Creating an anonymous placeholder
    """
    n_anon_placeholders = 0

    def __init__(self, *, consumers: Iterable[Node] = None, name: str = None):
        if name is None:
            name = f'{self.__class__.__name__}#{PlaceHolder.n_anon_placeholders}'
            PlaceHolder.n_anon_placeholders += 1

        super().__init__(consumers=consumers, name=name)

        try:
            ComputationalGraph._default_graph.placeholders.append(self)
        except (NameError, AttributeError):
            ComputationalGraph().as_default()
            ComputationalGraph._default_graph.placeholders.append(self)

    def __str__(self):
        """
        Returns the string representation of the node.

        Returns:
            str: The name of the node.
        """
        return f"cg.PlaceHolder(name='{self.name}')>"

    def __repr__(self):
        """
        Returns the string representation of the node.

        Returns:
            str: The name of the node.
        """
        return f"cg.PlaceHolder(name='{self.name}', #consumers = {len(self.consumers)})"


class Variable(Node):
    """
    Represents a trainable variable node in the computational graph.

    Variables are used to store and update trainable parameters during optimization.

    Parameters:
        value (Numeric): Initial value of the variable.
        trainable (bool): Indicates if the variable is trainable.
        consumers (Iterable[Node]): Nodes that consume the output of this variable.
        name (str): Name of the variable.

    Attributes:
        n_variables (int): Counter for variables.
        _graph (ComputationalGraph): The computational graph that the variable belongs to.

    Example:
        import compgraph as cg

        # Creating a trainable variable
        w = Variable(value=0.5, trainable=True, name='w')
        # Creating a trainable variable with default name
        b = Variable(value=1.0)
    """

    n_variables = 0

    def __init__(self,
                 value: Numeric = None,
                 trainable: bool = True,
                 *,
                 consumers: Iterable[Node] = None,
                 name: str = None):
        assert isinstance(value, (NumericTypes, Iterable)), (
            f'Value must be a Numeric type, one of '
            f'{", ".join(map(str, NumericTypes))} '
            f"or a Iterable that can be converted to a numpy array"
        )

        assert isinstance(trainable, bool), (
            "parameter `trainable` must be a boolean value"
        )

        if name is None:
            name = f'var#{Variable.n_variables}'
            Variable.n_variables += 1

        super().__init__(consumers=consumers, name=name)

        self.value = value if isinstance(
            value, NumericTypes) else np.array(value)
        self.trainable = trainable

        try:
            ComputationalGraph._default_graph.variables.append(self)
        except (NameError, AttributeError):
            ComputationalGraph().as_default()
            ComputationalGraph._default_graph.variables.append(self)

    def __str__(self):
        """
        Returns the string representation of the variable.
        """
        return f"cg.Variable(value={self.value}, name='{self.name}')"

    def __repr__(self):
        """
        Returns the string representation of the variable.
        """
        return f"cg.Variable(value={self.value}, name='{self.name}', #consumers = {len(self.consumers)})"


class Constant(Variable):
    """
    Represents a constant node in the computational graph.

    Constants are fixed values that do not get updated during optimization.

    Parameters:
        value (Numeric): Value of the constant.
        consumers (Iterable[Node]): Nodes that consume the output of this constant.
        name (str): Name of the constant.

    Attributes:
        _graph (ComputationalGraph): The computational graph that the constant belongs to.

    Example:
        import compgraph as cg

        pi = cg.Constant(value=3.14159, name='pi')  # Creating a named constant
        e = cg.Constant(value=2.71828)  # Creating a constant with default name
    """

    def __init__(self,
                 value: Numeric,
                 *,
                 consumers: Iterable[Node] = None,
                 name: str = None):
        if name is None:
            name = f'{value}'

        super().__init__(value, trainable=False, consumers=consumers, name=name)

        self.value = value

        try:
            ComputationalGraph._default_graph.constants.append(self)
        except (NameError, AttributeError):
            ComputationalGraph().as_default()
            ComputationalGraph._default_graph.constants.append(self)

        self._graph = ComputationalGraph._default_graph

    def __str__(self):
        """
        Returns the string representation of the variable.
        """
        return f"cg.Constant(value={self.value}, name='{self.name}')"

    def __repr__(self):
        """
        Returns the string representation of the variable.
        """
        return f"cg.Constant(value={self.value}, name='{self.name}', #consumers = {len(self.consumers)})"


class Operation(Node):
    """
    Represents an operation node in the computational graph.

    Operation nodes perform computations on input nodes to produce an output.

    Parameters:
        input_nodes (Iterable): Input nodes to the operation.
        consumers (Iterable[Node]): Nodes that consume the output of this operation.
        name (str): Name of the operation.

    Attributes:
        n_anon_operations (int): Counter for anonymous operations.
        _graph (ComputationalGraph): The computational graph that the operation belongs to.
        _op_name (str): Name of the operation.

    Example:
        from compgraph.graph.nodes import Operation

        add_op = Operation(input_nodes=[x, y], name='Addition')  # Creating an operation node
    """

    n_anon_operations = 0

    def __init__(self,
                 input_nodes: Iterable = None,
                 *,
                 consumers: Iterable[Node] = None,
                 name: str = None):
        if name is None:
            name = f'{self.__class__.__name__}#{Operation.n_anon_operations}'
            Operation.n_anon_operations += 1

        super().__init__(consumers=consumers, name=name)

        self.input_nodes: list = list(input_nodes) if input_nodes else list()

        for node in self.input_nodes:
            node.consumers.add(self)

        try:
            ComputationalGraph._default_graph.operations.append(self)
        except (NameError, AttributeError):
            ComputationalGraph().as_default()
            ComputationalGraph._default_graph.operations.append(self)

        self._graph = ComputationalGraph._default_graph

        self._op_name = self.__class__.__name__

    def __str__(self):
        """
        Returns the string representation of the variable.
        """
        if hasattr(self, 'value'):
            return f"cg.{self.__class__.__name__}(value={self.value}, name='{self.name}')"
        return f"cg.{self.__class__.__name__}(input_nodes={list(map(lambda x: x.name, self.input_nodes))}, name='{self.name}')"

    def __repr__(self):
        """
        Returns the string representation of the variable.
        """
        return f"cg.{self.__class__.__name__}(input_nodes={self.input_nodes}, name='{self.name}', #consumers = {len(self.consumers)})"

    def __call__(self):
        """
        Must be implemented for particular operations.

        This method defines the logic of the operation and is called during graph execution.

        Raises:
            NotImplementedError: If the operation logic is not specified.
        """
        raise NotImplementedError("operation logic is not specified")

    def gradient(self, grad):
        """
        Must be implemented for particular operations.

        This method defines the gradient of the operation and is called during graph execution.

        Raises:
            NotImplementedError: If the operation logic is not specified.
        """
        raise NotImplementedError("gradient logic is not specified")


class add(Operation):
    """
    Represents a node that performs addition in the computational graph.

    Parameters:
        input_node1 (Node): First input node.
        input_node2 (Node): Second input node.
        consumers (Iterable[Node]): Nodes that consume the output of this addition node.
        name (str): Name of the addition node.

    Example:
        import compgraph as cg

        add_node = cg.add(x, y, name='x + y')  # Creating an addition node
    """

    def __init__(self,
                 input_node1: Node,
                 input_node2: Node,
                 *,
                 consumers: Iterable[Node] = None,
                 name: str = None):
        if name is None:
            if any(op in input_node1.name for op in ['+', '-', '*', '/', '@', '**']):
                input1_name = f'({input_node1.name})'
            else:
                input1_name = f'{input_node1.name}'

            if any(op in input_node2.name for op in ['+', '-', '*', '/', '@', '**']):
                input2_name = f'({input_node2.name})'
            else:
                input2_name = f'{input_node2.name}'

            name = f'{input1_name} + {input2_name}'

        super().__init__(
            [input_node1, input_node2],
            consumers=consumers,
            name=name
        )

    def __call__(self,
                 input1: Numeric,
                 input2: Numeric) -> Numeric:
        """
        Perform addition on the input values.

        Parameters:
            input1 (Numeric): First input value.
            input2 (Numeric): Second input value.

        Returns:
            Numeric: Result of the addition operation.
        """
        return np.add(input1, input2)

    def gradient(self, grad: Numeric) -> np.ndarray:
        """
        Compute the gradient of the addition operation.

        Parameters:
            grad (Numeric): Gradient of the output node.

        Returns:
            np.ndarray: Gradients with respect to the input nodes.
        """

        a = self.input_nodes[0].output
        b = self.input_nodes[1].output

        grad_wrt_a = grad
        grad_wrt_b = grad

        #
        # The following becomes relevant if a and b are of different shapes.
        #
        while np.ndim(grad_wrt_a) > len(a.shape):
            grad_wrt_a = np.sum(grad_wrt_a, axis=0)

        for axis, size in enumerate(a.shape):
            if size == 1:
                grad_wrt_a = np.sum(grad_wrt_a, axis=axis, keepdims=True)

        while np.ndim(grad_wrt_b) > len(b.shape):
            grad_wrt_b = np.sum(grad_wrt_b, axis=0)

        for axis, size in enumerate(b.shape):
            if size == 1:
                grad_wrt_b = np.sum(grad_wrt_b, axis=axis, keepdims=True)

        return [grad_wrt_a, grad_wrt_b]


class subtract(Operation):
    """
    Represents a node that performs subtraction in the computational graph.

    Parameters:
        input_node1 (Node): First input node.
        input_node2 (Node): Second input node.
        consumers (Iterable[Node]): Nodes that consume the output of this subtraction node.
        name (str): Name of the subtraction node.

    Example:
        import compgraph as cg

        subtract_node = cg.subtract(x, y, name='x - y')  # Creating a subtraction node
    """

    def __init__(self,
                 input_node1: Node,
                 input_node2: Node,
                 *,
                 consumers: Iterable[Node] = None,
                 name: str = None):
        if name is None:
            if any(op in input_node1.name for op in ['+', '-', '*', '/', '@', '**']):
                input1_name = f'({input_node1.name})'
            else:
                input1_name = f'{input_node1.name}'

            if any(op in input_node2.name for op in ['+', '-', '*', '/', '@', '**']):
                input2_name = f'({input_node2.name})'
            else:
                input2_name = f'{input_node2.name}'

            name = f'{input1_name} - {input2_name}'

        super().__init__(
            [input_node1, input_node2],
            consumers=consumers,
            name=name
        )

    def __call__(self,
                 input1: Numeric,
                 input2: Numeric) -> Numeric:
        """
        Perform subtraction on the input values.

        Parameters:
            input1 (Numeric): First input value.
            input2 (Numeric): Second input value.

        Returns:
            Numeric: Result of the subtraction operation.
        """
        return np.subtract(input1, input2)

    def gradient(self, grad: Numeric) -> np.ndarray:
        """
        Compute the gradient of the subtraction operation.

        Parameters:
            grad (Numeric): Gradient of the output node.

        Returns:
            np.ndarray: Gradients with respect to the input nodes.
        """
        a = self.input_nodes[0].output
        b = self.input_nodes[1].output

        grad_wrt_a = grad
        grad_wrt_b = grad

        #
        # The following becomes relevant if a and b are of different shapes.
        #
        while np.ndim(grad_wrt_a) > len(a.shape):
            grad_wrt_a = np.sum(grad_wrt_a, axis=0)

        for axis, size in enumerate(a.shape):
            if size == 1:
                grad_wrt_a = np.sum(grad_wrt_a, axis=axis, keepdims=True)

        while np.ndim(grad_wrt_b) > len(b.shape):
            grad_wrt_b = np.sum(grad_wrt_b, axis=0)

        for axis, size in enumerate(b.shape):
            if size == 1:
                grad_wrt_b = np.sum(grad_wrt_b, axis=axis, keepdims=True)

        return np.array([grad_wrt_a, -grad_wrt_b])


class multiply(Operation):
    """
    Represents a node that performs multiplication in the computational graph.

    Parameters:
        input_node1 (Node): First input node.
        input_node2 (Node): Second input node.
        consumers (Iterable[Node]): Nodes that consume the output of this multiplication node.
        name (str): Name of the multiplication node.

    Example:
        import compgraph as cg

        multiply_node = cg.multiply(x, y, name='x * y')  # Creating a multiplication node
    """

    def __init__(self,
                 input_node1: Node,
                 input_node2: Node,
                 *,
                 consumers: Iterable[Node] = None,
                 name: str = None):
        if name is None:
            if any(op in input_node1.name for op in ['+', '-', '*', '/', '@', '**']):
                input1_name = f'({input_node1.name})'
            else:
                input1_name = f'{input_node1.name}'

            if any(op in input_node2.name for op in ['+', '-', '*', '/', '@', '**']):
                input2_name = f'({input_node2.name})'
            else:
                input2_name = f'{input_node2.name}'

            name = f'{input1_name} * {input2_name}'

        super().__init__(
            [input_node1, input_node2],
            consumers=consumers,
            name=name
        )

    def __call__(self,
                 input1: Numeric,
                 input2: Numeric) -> Numeric:
        """
        Perform multiplication on the input values.

        Parameters:
            input1 (Numeric): First input value.
            input2 (Numeric): Second input value.

        Returns:
            Numeric: Result of the multiplication operation.
        """
        return np.multiply(input1, input2)

    def gradient(self, grad: Numeric) -> np.ndarray:
        """
        Compute the gradient of the multiplication operation.

        Parameters:
            grad (Numeric): Gradient of the output node.

        Returns:
            np.ndarray: Gradients with respect to the input nodes.
        """
        return np.array([grad * self.input_nodes[1].output, grad * self.input_nodes[0].output])


class divide(Operation):
    """
    Represents a node that performs division in the computational graph.

    Parameters:
        input_node1 (Node): First input node.
        input_node2 (Node): Second input node.
        consumers (Iterable[Node]): Nodes that consume the output of this division node.
        name (str): Name of the division node.

    Example:
        import compgraph as cg

        divide_node = cg.divide(x, y, name='x / y')  # Creating a division node
    """

    def __init__(self,
                 input_node1: Node,
                 input_node2: Node,
                 *,
                 consumers: Iterable[Node] = None,
                 name: str = None):
        if name is None:
            if any(op in input_node1.name for op in ['+', '-', '*', '/', '@', '**']):
                input1_name = f'({input_node1.name})'
            else:
                input1_name = f'{input_node1.name}'

            if any(op in input_node2.name for op in ['+', '-', '*', '/', '@', '**']):
                input2_name = f'({input_node2.name})'
            else:
                input2_name = f'{input_node2.name}'

            name = f'{input1_name} / {input2_name}'

        super().__init__(
            [input_node1, input_node2],
            consumers=consumers,
            name=name
        )

    def __call__(self,
                 input1: Numeric,
                 input2: Numeric) -> Numeric:
        """
        Perform division on the input values.

        Parameters:
            input1 (Numeric): First input value.
            input2 (Numeric): Second input value.

        Returns:
            Numeric: Result of the division operation.
        """
        return np.divide(input1, input2)

    def gradient(self, grad):
        """
        Compute the gradient of the division operation.

        Parameters:
            grad (Numeric): Gradient of the output node.

        Returns:
            np.ndarray: Gradients with respect to the input nodes.
        """
        return np.array(
            [
                grad / (_ := self.input_nodes[1].output),
                -grad * self.input_nodes[0].output / (_ ** 2)
            ]
        )


class power(Operation):
    """
    Represents a node that performs exponentiation in the computational graph.

    Parameters:
        input_node1 (Node): Base input node.
        input_node2 (Node): Exponent input node.
        consumers (Iterable[Node]): Nodes that consume the output of this power node.
        name (str): Name of the power node.

    Example:
        import compgraph as cg

        power_node = cg.power(x, y, name='x ** y')  # Creating a power node
    """

    def __init__(self,
                 input_node1: Node,
                 input_node2: Node,
                 *,
                 consumers: Iterable[Node] = None,
                 name: str = None):
        if name is None:
            if any(op in input_node1.name for op in ['+', '-', '*', '/', '@', '**']):
                input1_name = f'({input_node1.name})'
            else:
                input1_name = f'{input_node1.name}'

            if any(op in input_node2.name for op in ['+', '-', '*', '/', '@', '**']):
                input2_name = f'({input_node2.name})'
            else:
                input2_name = f'{input_node2.name}'

            name = f'{input1_name} ** {input2_name}'

        super().__init__(
            [input_node1, input_node2],
            consumers=consumers,
            name=name
        )

    def __call__(self,
                 input1: Numeric,
                 input2: Numeric) -> Numeric:
        """
        Perform exponentiation on the input values.

        Parameters:
            input1 (Numeric): Base input value.
            input2 (Numeric): Exponent input value.

        Returns:
            Numeric: Result of the exponentiation operation.
        """
        return np.power(input1, input2)

    def gradient(self, grad):
        """
        Compute the gradient of the exponentiation operation.

        Parameters:
            grad (Numeric): Gradient of the output node.

        Returns:
            np.ndarray: Gradients with respect to the input nodes.
        """
        x, y = map(lambda x: x.output, self.input_nodes)
        return np.array(
            [
                grad * y * (x ** (y - 1)),
                grad * self.output * np.log(x)
            ]
        )


class matmul(Operation):
    """
    Represents a node that performs matrix multiplication in the computational graph.

    Parameters:
        input_node1 (Node): First input node.
        input_node2 (Node): Second input node.
        consumers (Iterable[Node]): Nodes that consume the output of this matrix multiplication node.
        name (str): Name of the matrix multiplication node.

    Example:
        import compgraph as cg

        matmul_node = cg.matmul(x, y, name='x @ y')  # Creating a matrix multiplication node
    """

    def __init__(self,
                 input_node1: Node,
                 input_node2: Node,
                 *,
                 consumers: Iterable[Node] = None,
                 name: str = None):
        if name is None:
            if any(op in input_node1.name for op in ['+', '-', '*', '/', '@', '**']):
                input1_name = f'({input_node1.name})'
            else:
                input1_name = f'{input_node1.name}'

            if any(op in input_node2.name for op in ['+', '-', '*', '/', '@', '**']):
                input2_name = f'({input_node2.name})'
            else:
                input2_name = f'{input_node2.name}'

            name = f'{input1_name} @ {input2_name}'

        super().__init__(
            [input_node1, input_node2],
            consumers=consumers,
            name=name
        )

    def __call__(self,
                 input1: Numeric,
                 input2: Numeric) -> Numeric:
        """
        Perform matrix multiplication on the input values.

        Parameters:
            input1 (Numeric): First input value.
            input2 (Numeric): Second input value.

        Returns:
            Numeric: Result of the matrix multiplication operation.
        """
        try:
            return input1 @ input2
        except TypeError:
            return np.dot(input1, input2)

    def gradient(self, grad: Numeric) -> np.ndarray:
        """
        Compute the gradient of the matrix multiplication operation.

        Parameters:
            grad (Numeric): Gradient of the output node.

        Returns:
            np.ndarray: Gradients with respect to the input nodes.
        """
        return (
            [
                grad.dot(self.input_nodes[1].output.T),
                self.input_nodes[0].output.T.dot(grad)
            ]
        )


class negative(Operation):
    """
    Represents a node that computes the negative of a value in the computational graph.

    Parameters:
        input_node1 (Node): Input node.
        consumers (Iterable[Node]): Nodes that consume the output of this negative node.
        name (str): Name of the negative node.

    Example:
        import compgraph as cg

        negative_node = cg.negative(x, name='-x')  # Creating a negative node
    """

    def __init__(self,
                 input_node1: Node,
                 *,
                 consumers: Iterable[Node] = None,
                 name: str = None):
        super().__init__(
            [input_node1],
            consumers=consumers,
            name=name or f'-({input_node1.name})'
        )

    def __call__(self, input1: Numeric) -> Numeric:
        """
        Compute the negative of the input value.

        Parameters:
            input1 (Numeric): Input value.

        Returns:
            Numeric: Negative of the input value.
        """
        return np.negative(input1)

    def gradient(self, grad: Numeric) -> np.ndarray:
        """
        Compute the gradient of the negative operation.

        Parameters:
            grad (Numeric): Gradient of the output node.

        Returns:
            np.ndarray: Gradients with respect to the input nodes.
        """
        return self(grad)
