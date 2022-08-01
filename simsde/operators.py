import sympy
import symnum.numpy as snp

def diff(expr, wrt, order=1):
    """Compute the `order`th derivative of expression `expr` with respect to variable(s) `wrt`.
    
    Compared to base SymPy implementation this makes two changes
    
      * SymPy orders the dimensions in the returned derivative array with the dimensions
        corresponding to those of `wrt` first and those of `expr` last. Here this order
        is transposes such that the `expr` dimensions come first, which is consistent
        with the standard 'numerator' layout of Jacobians 
        (https://en.wikipedia.org/wiki/Matrix_calculus#Layout_conventions)
      * SymPy will return zeros of the shape of `expr` if none of the terms in `expr` depend
        on the variables in `wrt` rather than a zero array with total number of dimensions
        equal to the sum of the number of dimensions of `expr` and `wrt`.
    """
    deriv = expr.diff(wrt, order)
    expr_shape = expr.shape if hasattr(expr, "shape") else ()
    wrt_shape = wrt.shape if hasattr(wrt, "shape") else ()
    deriv_shape = deriv.shape if hasattr(deriv, "shape") else ()
    expected_shape = wrt_shape * order + expr_shape
    if deriv_shape == expected_shape:
        transpose_permutation = (
            tuple(range(len(wrt_shape) * order, len(deriv_shape))) + 
            tuple(range(len(wrt_shape) * order))
        )
        return deriv if deriv_shape == () else deriv.transpose(transpose_permutation)
    else:
        if (deriv_shape == () and deriv == 0) or all(deriv == snp.zeros(deriv_shape)):
            return snp.zeros(expr_shape + wrt_shape * order)
        else:
            raise ValueError(
                f"Unexpected derivative shape: {deriv_shape}, expected {expected_shape}"
            )


def v_hat_k(drift_func, diff_coeff, k, dim_r=None):
    
    def v_hat_0_func(func):
        
        def v_hat_0_func_x_z(x, z):
            dim_r_ = x.shape[0] if dim_r is None else dim_r
            a = drift_func(x, z)
            B = diff_coeff(x, z)
            BB_T = B @ B.T
            f = func(x, z)
            df_dx = diff(f, x)
            d2f_dx2 = diff(f, x[:dim_r_], 2)
            return df_dx @ a + (d2f_dx2 * BB_T).sum((1, 2)) / 2
        
        return v_hat_0_func_x_z
    
    def v_hat_k_func(func):
        
        def v_hat_k_func_x_z(x, z):
            dim_r_ = x.shape[0] if dim_r is None else dim_r
            B = diff_coeff(x, z)
            f = func(x, z)
            df_dx = diff(f, x[:dim_r_])
            return df_dx @ B[:, k - 1]
        
        return v_hat_k_func_x_z
    
    if k == 0:
        return v_hat_0_func
    else:
        return v_hat_k_func


def subscript_k(func, k):
    
    def func_k(x, z):
        return func(x, z)[k]
    
    return func_k


def square_bracket(drift_func, diff_coeff, k_1, k_2):

    v_k_1 = drift_func if k_1 == 0 else subscript_k(diff_coeff, k_1 - 1)
    v_k_2 = drift_func if k_2 == 0 else subscript_k(diff_coeff, k_2 - 1)

    def square_bracket_v_k_1_v_k_2(x, z):
        return (
            v_hat_k(drift_func, diff_coeff, k_1)(v_k_2)(x, z) 
            - v_hat_k(drift_func, diff_coeff, k_2)(v_k_1)(x, z)
        )

    return square_bracket_v_k_1_v_k_2