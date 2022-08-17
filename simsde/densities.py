import sympy
import symnum.numpy as snp
from simsde.operators import v_hat_k


def local_gaussian_mean_and_covariance(drift_func_rough, drift_func_smooth, diff_coeff_rough):

    def drift_func(x, θ):
        return snp.concatenate((drift_func_rough(x, θ), drift_func_smooth(x, θ)))

    def mean_and_covariance(x, θ, t):
        dim_r = drift_func_rough(x, θ).shape[0]
        x_r, x_s = x[:dim_r], x[dim_r:]
        μ = snp.concatenate(
            [
                x_r + drift_func_rough(x, θ) * t,
                x_s + drift_func_smooth(x, θ) * t
                + v_hat_k(drift_func, diff_coeff_rough, 0, dim_r)(drift_func_smooth)(x, θ) * t**2 / 2
            ]
        )
        B_r = diff_coeff_rough(x, θ)
        C_s = snp.array(
            [
                v_hat_k(drift_func, diff_coeff_rough, k, dim_r)(drift_func_smooth)(x, θ)
                for k in range(1, dim_r + 1)
            ],
        ).T
        Σ_11 = B_r @ B_r.T * t
        Σ_12 = B_r @ C_s.T * t**2 / 2
        Σ_22 = C_s @ C_s.T * t**3 / 3
        Σ = snp.concatenate(
            [
                snp.concatenate([Σ_11, Σ_12], axis=1),
                snp.concatenate([Σ_12.T, Σ_22], axis=1)
            ],
            axis=0
        )
        return μ, Σ

    return mean_and_covariance


def local_gaussian_log_transition_density(
    drift_func_rough, drift_func_smooth, diff_coeff_rough, max_dimension=3
):

    mean_and_covariance = local_gaussian_mean_and_covariance(
        drift_func_rough, drift_func_smooth, diff_coeff_rough
    )

    def log_transition_density(x_t, x_0, θ, t):
        dim_x = x_0.shape[0]
        μ, Σ = mean_and_covariance(x_0, θ, t)
        if dim_x > max_dimension:
            raise ValueError(
                f"x has dimension dim_x={dim_x} which is greater than max_dimension={max_dimension}. "
                f"Symbolically computing the log density requires evaluating the determinant of the "
                f"(dim_x, dim_x) covariance matrix with a cost factorial in dim_x. To allow evaluation "
                f"increase max_dimension to be more than or equal to dim_x but be aware this may lead "
                f"to very long evaluation times"
            )
        Σ = sympy.Matrix(Σ)
        return -(
            (x_t - μ) * Σ.inv() * (x_t - μ) / 2
            + snp.log(Σ.det()) / 2
            + snp.log(2 * snp.pi) * (dim_x / 2)
        )

    return log_transition_density
