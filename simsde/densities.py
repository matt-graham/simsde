import sympy
import symnum.numpy as snp
from simsde.operators import v_hat_k, subscript_k


def local_gaussian_mean_and_covariance(
    drift_func_rough, drift_func_smooth, diff_coeff_rough
):
    def drift_func(x, θ):
        return snp.concatenate((drift_func_rough(x, θ), drift_func_smooth(x, θ)))

    def mean_and_covariance(x, θ, t):
        dim_r = drift_func_rough(x, θ).shape[0]
        x_r, x_s = x[:dim_r], x[dim_r:]
        μ = snp.concatenate(
            [
                x_r + drift_func_rough(x, θ) * t,
                x_s
                + drift_func_smooth(x, θ) * t
                + v_hat_k(drift_func, diff_coeff_rough, 0, dim_r)(drift_func_smooth)(
                    x, θ
                )
                * t**2
                / 2,
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
                snp.concatenate([Σ_12.T, Σ_22], axis=1),
            ],
            axis=0,
        )
        return μ, Σ

    return mean_and_covariance


def local_gaussian_log_transition_density(
    drift_func_rough, drift_func_smooth, diff_coeff_rough
):

    mean_and_covariance = local_gaussian_mean_and_covariance(
        drift_func_rough, drift_func_smooth, diff_coeff_rough
    )

    def log_transition_density(x_t, x_0, θ, t):
        dim_x = x_0.shape[0]
        μ, Σ = mean_and_covariance(x_0, θ, t)
        Σ = sympy.Matrix(Σ)
        chol_Σ = Σ.cholesky(hermitian=False)
        x_t_minus_μ = sympy.Matrix(x_t - μ)
        return -(
            (
                x_t_minus_μ.T
                * chol_Σ.T.upper_triangular_solve(
                    chol_Σ.lower_triangular_solve(x_t_minus_μ)
                )
            )[0, 0]
            / 2
            + snp.log(chol_Σ.diagonal()).sum()
            + snp.log(2 * snp.pi) * (dim_x / 2)
        )

    return log_transition_density


def improved_scheme_log_transition_density(
    drift_func_rough, drift_func_smooth, diff_coeff_rough
):
    def drift_func(x, θ):
        return snp.concatenate((drift_func_rough(x, θ), drift_func_smooth(x, θ)))

    def outer(a, b=None):
        b = a if b is None else b
        return a.reshape((a.shape[0], 1)) * b.reshape((1, b.shape[0]))

    mean_and_covariance = local_gaussian_mean_and_covariance(
        drift_func_rough, drift_func_smooth, diff_coeff_rough
    )

    def Φ_2(t, x, y, θ):
        _, Σ_1 = mean_and_covariance(x, θ, 1)
        dim_r = drift_func_rough(x, θ).shape[0]
        x_r, x_s = x[:dim_r], x[dim_r:]
        y_r, y_s = y[:dim_r], y[dim_r:]
        v_r = [drift_func_rough] + [
            subscript_k(diff_coeff_rough, k) for k in range(dim_r)
        ]
        Σ_1_h = snp.concatenate(
            (
                (y_r - x_r - drift_func_smooth(x, θ) * t) / snp.sqrt(t),
                (
                    y_s
                    - x_s
                    - drift_func_smooth(x, θ) * t
                    - v_hat_k(drift_func, diff_coeff_rough, 0, dim_r)(
                        drift_func_smooth
                    )(x, θ)
                    * t**2
                    / 2
                )
                / snp.sqrt(t) ** 3,
            )
        )
        M_rr = (t / 2) * sum(
            [
                outer(
                    v_hat_k(drift_func, diff_coeff_rough, k, dim_r)(drift_func_rough)(
                        x, θ
                    )
                    + v_hat_k(drift_func, diff_coeff_rough, 0, dim_r)(v_r[k])(x, θ),
                    v_r[k](x, θ),
                )
                for k in range(1, dim_r + 1)
            ]
        ) + (t / 4) * sum(
            [
                outer(
                    v_hat_k(drift_func, diff_coeff_rough, k_1, dim_r)(v_r[k_2])(x, θ),
                    v_hat_k(drift_func, diff_coeff_rough, k_2, dim_r)(v_r[k_1])(x, θ),
                )
                for k_1 in range(1, dim_r + 1)
                for k_2 in range(1, dim_r + 1)
            ]
        )
        M_rs = (
            t
            * sum(
                [
                    outer(
                        v_hat_k(drift_func, diff_coeff_rough, k, dim_r)(
                            drift_func_rough
                        )(x, θ)
                        / 3
                        + v_hat_k(drift_func, diff_coeff_rough, 0, dim_r)(v_r[k])(x, θ)
                        / 6,
                        v_hat_k(drift_func, diff_coeff_rough, k, dim_r)(
                            drift_func_smooth
                        )(x, θ),
                    )
                    for k in range(1, dim_r)
                ]
            )
            + (t / 6)
            * sum(
                [
                    outer(
                        v_r[k](x, θ),
                        v_hat_k(drift_func, diff_coeff_rough, 0, dim_r)(
                            v_hat_k(drift_func, diff_coeff_rough, k, dim_r)(
                                drift_func_smooth
                            )
                        )(x, θ)
                        + v_hat_k(drift_func, diff_coeff_rough, k, dim_r)(
                            v_hat_k(drift_func, diff_coeff_rough, 0, dim_r)(
                                drift_func_smooth
                            )
                        )(x, θ),
                    )
                    for k in range(1, dim_r + 1)
                ]
            )
            + (t / 6)
            * sum(
                [
                    outer(
                        v_hat_k(drift_func, diff_coeff_rough, k_1, dim_r)(v_r[k_2])(
                            x, θ
                        ),
                        v_hat_k(drift_func, diff_coeff_rough, k_1, dim_r)(
                            v_hat_k(drift_func, diff_coeff_rough, k_2, dim_r)(
                                drift_func_smooth
                            )
                        )(x, θ),
                    )
                    for k_1 in range(1, dim_r + 1)
                    for k_2 in range(1, dim_r + 1)
                ]
            )
        )
        M_ss = t * sum(
            [
                outer(
                    v_hat_k(drift_func, diff_coeff_rough, k, dim_r)(drift_func_smooth)(
                        x, θ
                    ),
                    (
                        v_hat_k(drift_func, diff_coeff_rough, 0, dim_r)(
                            v_hat_k(drift_func, diff_coeff_rough, k, dim_r)(
                                drift_func_smooth
                            )
                        )(x, θ)
                        / 6
                        + v_hat_k(drift_func, diff_coeff_rough, k, dim_r)(
                            v_hat_k(drift_func, diff_coeff_rough, 0, dim_r)(
                                drift_func_smooth
                            )
                        )(x, θ)
                        / 8
                    ),
                )
                for k in range(1, dim_r + 1)
            ]
        ) + (t / 24) * sum(
            [
                outer(
                    v_hat_k(drift_func, diff_coeff_rough, k_1, dim_r)(
                        v_hat_k(drift_func, diff_coeff_rough, k_2, dim_r)(
                            drift_func_smooth
                        )
                    )(x, θ)
                )
                for k_1 in range(1, dim_r + 1)
                for k_2 in range(1, dim_r + 1)
            ]
        )
        h = sympy.Matrix(Σ_1).cholesky_solve(sympy.Matrix(Σ_1_h))
        H = h * h.T - sympy.Matrix(Σ_1).inverse_CH()
        return (
            (M_rr * H[:dim_r, :dim_r]).sum()
            + (M_rs * H[:dim_r, dim_r:]).sum()
            + (M_ss * H[dim_r:, dim_r:]).sum()
        )

    def log_transition_density(x_t, x_0, θ, t):
        dim_x = x_0.shape[0]
        μ, Σ = mean_and_covariance(x_0, θ, t)
        Σ = sympy.Matrix(Σ)
        chol_Σ = Σ.cholesky(hermitian=False)
        x_t_minus_μ = sympy.Matrix(x_t - μ)
        return -(
            (
                x_t_minus_μ.T
                * chol_Σ.T.upper_triangular_solve(
                    chol_Σ.lower_triangular_solve(x_t_minus_μ)
                )
            )[0, 0]
            / 2
            + snp.log(chol_Σ.diagonal()).sum()
            + snp.log(2 * snp.pi) * (dim_x / 2)
            - Φ_2(t, x_0, x_t, θ)
        )

    return log_transition_density
