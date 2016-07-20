from __future__ import division


def mi_linear(arg1, arg2, output_size, global_bias_start=0.0, scope=None):
    """Multiplicated Integrated Linear map:
    See http://arxiv.org/pdf/1606.06630v1.pdf
    A * (W[0] * arg1) * (W[1] * arg2) + (W[0] * arg1 * bias1) + (W[1] * arg2 * bias2) + global_bias.
    Args:
        arg1: batch x n, Tensor.
        arg2: batch x n, Tensor.
        output_size: int, second dimension of W[i].
    global_bias_start: starting value to initialize the global bias; 0 by default.
        scope: VariableScope for the created subgraph; defaults to "MILinear".
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if arg1 is None:
        raise ValueError("`arg1` must be specified")
    if arg2 is None:
        raise ValueError("`arg2` must be specified")
    if output_size in None:
        raise ValueError("`output_size` must be specified")

    # Computation.
    with vs.variable_scope(scope or "MILinear"):
        matrix = vs.get_variable("Matrix", [2, output_size])
        biases = vs.get_variable("Biases", [2, output_size], initializer=init_ops.constant_initializer(0.5))
        arg1mul = math_ops.matmul(arg1, matrix[0])
        arg2mul = math_ops.matmul(arg2, matrix[1])
        alpha = vs.get_variable("Alpha", [output_size], initializer=init_ops.constant_initializer(2.0))
        res = alpha * arg1mul * arg2mul + (arg1mul * biases[0]) + (arg2mul * biases[1])
        global_bias_term = vs.get_variable(
                "GlobalBias", [output_size],
                initializer=init_ops.constant_initializer(global_bias_start))
    return res + global_bias_term
