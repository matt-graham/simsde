import symnum.numpy as snp


def diff(expression, wrt, order=1):
    """Compute the `order`th derivative of `expression` with respect to variable(s) `wrt`.

    Compared to base SymPy implementation this makes two changes

      * SymPy orders the dimensions in the returned derivative array with the dimensions
        corresponding to those of `wrt` first and those of `expression` last. Here this
        order is transposed such that the `expression` dimensions come first, which is
        consistent with the standard 'numerator' layout of Jacobians
        (https://en.wikipedia.org/wiki/Matrix_calculus#Layout_conventions).
      * SymPy will return zeros of the shape of `expression` if none of the terms in
        `expression` depend on the variables in `wrt` rather than a zero array with
        total number of dimensions equal to the sum of the number of dimensions of
        `expression` and `wrt`, which is the behaviour here.
    """
    derivative = expression.diff(wrt, order)
    expression_shape = expression.shape if hasattr(expression, "shape") else ()
    wrt_shape = wrt.shape if hasattr(wrt, "shape") else ()
    derivative_shape = derivative.shape if hasattr(derivative, "shape") else ()
    expected_shape = wrt_shape * order + expression_shape
    if derivative_shape == expected_shape:
        transpose_permutation = tuple(
            range(len(wrt_shape) * order, len(derivative_shape))
        ) + tuple(range(len(wrt_shape) * order))
        return (
            derivative
            if derivative_shape == ()
            else derivative.transpose(transpose_permutation)
        )
    else:
        if (derivative_shape == () and derivative == 0) or all(
            derivative == snp.zeros(derivative_shape)
        ):
            return snp.zeros(expression_shape + wrt_shape * order)
        else:
            raise ValueError(
                f"Got derivative shape: {derivative_shape}, expected {expected_shape}"
            )


def v_hat_k(drift_func, diff_coeff, k, dim_r=None):
    def v_hat_0_func(func):
        def v_hat_0_func_x_θ(x, θ):
            dim_r_ = x.shape[0] if dim_r is None else dim_r
            a = drift_func(x, θ)
            B = diff_coeff(x, θ)
            BB_T = B @ B.T
            f = func(x, θ)
            df_dx = diff(f, x)
            d2f_dx2 = diff(f, x[:dim_r_], 2)
            return df_dx @ a + (d2f_dx2 * BB_T).sum((1, 2)) / 2

        return v_hat_0_func_x_θ

    def v_hat_k_func(func):
        def v_hat_k_func_x_θ(x, θ):
            dim_r_ = x.shape[0] if dim_r is None else dim_r
            B = diff_coeff(x, θ)
            f = func(x, θ)
            df_dx = diff(f, x[:dim_r_])
            return df_dx @ B[:, k - 1]

        return v_hat_k_func_x_θ

    if k == 0:
        return v_hat_0_func
    else:
        return v_hat_k_func


def subscript_k(func, k):
    def func_k(x, θ):
        return func(x, θ)[k]

    return func_k


def square_bracket(drift_func, diff_coeff, k_1, k_2):

    v_k_1 = drift_func if k_1 == 0 else subscript_k(diff_coeff, k_1 - 1)
    v_k_2 = drift_func if k_2 == 0 else subscript_k(diff_coeff, k_2 - 1)

    def square_bracket_v_k_1_v_k_2(x, θ):
        return v_hat_k(drift_func, diff_coeff, k_1)(v_k_2)(x, θ) - v_hat_k(
            drift_func, diff_coeff, k_2
        )(v_k_1)(x, θ)

    return square_bracket_v_k_1_v_k_2
