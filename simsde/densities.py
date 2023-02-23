import sympy
import symnum.numpy as snp
from simsde.operators import v_hat_k, subscript_k

# This is the local gaussian density for the model Hypo-I.
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


# This return mean and covariance of the local gaussian density for the model Hypo-II.
# Note: drift_func_smooth_1 is assumed to be the drift for the most smooth components and does not depend on rough components
def local_gaussian_mean_and_covariance_II(
    drift_func_smooth_1, drift_func_smooth_2, drift_func_rough, diff_coeff_rough
):
    def drift_func(x, θ):
        return snp.concatenate(
            (
                drift_func_rough(x, θ),
                drift_func_smooth_2(x, θ),
                drift_func_smooth_1(x, θ),
            )
        )

    def mean_and_covariance(x, θ, t):
        dim_r = drift_func_rough(x, θ).shape[0]
        dim_s2 = drift_func_smooth_2(x, θ).shape[0]
        x_r, x_s_2, x_s_1 = x[:dim_r], x[dim_r : dim_r + dim_s2], x[dim_r + dim_s2 :]
        μ = snp.concatenate(
            [
                # drift approximation for the rough component up to O (t) terms
                x_r + drift_func_rough(x, θ) * t,
                # drift approximation for the second smooth component (position in GLE)
                # up to O (t^2) terms
                x_s_2
                + drift_func_smooth_2(x, θ) * t
                + v_hat_k(drift_func, diff_coeff_rough, 0, dim_r)(drift_func_smooth_2)(
                    x, θ
                )
                * t**2
                / 2,
                # drift approximation for the first smooth component (position in GLE)
                # up to O (t^3) terms
                x_s_1
                + drift_func_smooth_1(x, θ) * t
                + v_hat_k(drift_func, diff_coeff_rough, 0, dim_r)(drift_func_smooth_1)(
                    x, θ
                )
                * t**2
                / 2
                + v_hat_k(drift_func, diff_coeff_rough, 0, dim_r)(
                    v_hat_k(drift_func, diff_coeff_rough, 0, dim_r)(drift_func_smooth_1)
                )(x, θ)
                * t**3
                / 6,
            ]
        )
        C_r = diff_coeff_rough(x, θ)
        C_s2 = snp.array(
            [
                v_hat_k(drift_func, diff_coeff_rough, k, dim_r)(drift_func_smooth_2)(
                    x, θ
                )
                for k in range(1, dim_r + 1)
            ],
        ).T
        C_s1 = snp.array(
            [
                v_hat_k(drift_func, diff_coeff_rough, k, dim_r)(
                    v_hat_k(drift_func, diff_coeff_rough, 0, dim_r)(drift_func_smooth_1)
                )(x, θ)
                for k in range(1, dim_r + 1)
            ],
        ).T
        Σ_RR = C_r @ C_r.T * t
        Σ_RS2 = C_r @ C_s2.T * t**2 / 2
        Σ_RS1 = C_r @ C_s1.T * t**3 / 6
        Σ_S2S2 = C_s2 @ C_s2.T * t**3 / 3
        Σ_S2S1 = C_s2 @ C_s1.T * t**4 / 8
        Σ_S1S1 = C_s1 @ C_s1.T * t**5 / 20
        Σ = snp.concatenate(
            [
                snp.concatenate([Σ_RR, Σ_RS2, Σ_RS1], axis=1),
                snp.concatenate([Σ_RS2.T, Σ_S2S2, Σ_S2S1], axis=1),
                snp.concatenate([Σ_RS1.T, Σ_S2S1.T, Σ_S1S1], axis=1),
            ],
            axis=0,
        )
        return μ, Σ

    return mean_and_covariance


# This is the gaussian density with higher order approximation for the drift function
# for the model Hypo-I
def local_gaussian_improved_mean_and_covariance(
    drift_func_rough, drift_func_smooth, diff_coeff_rough
):
    def drift_func(x, θ):
        return snp.concatenate((drift_func_rough(x, θ), drift_func_smooth(x, θ)))

    def mean_and_covariance(x, θ, t):
        dim_r = drift_func_rough(x, θ).shape[0]
        x_r, x_s = x[:dim_r], x[dim_r:]
        μ = snp.concatenate(
            [
                x_r
                + drift_func_rough(x, θ) * t
                + v_hat_k(drift_func, diff_coeff_rough, 0, dim_r)(drift_func_rough)(
                    x, θ
                )
                * t**2
                / 2,  # drift approximation for rough components with the additional term O(t^2)
                x_s
                + drift_func_smooth(x, θ) * t
                + v_hat_k(drift_func, diff_coeff_rough, 0, dim_r)(drift_func_smooth)(
                    x, θ
                )
                * t**2
                / 2
                + v_hat_k(drift_func, diff_coeff_rough, 0, dim_r)(
                    v_hat_k(drift_func, diff_coeff_rough, 0, dim_r)(drift_func_smooth)
                )(x, θ)
                * t**3
                / 6,  # drift approximation for smooth components with the additional term O(t^3)
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


def improved_scheme_correction_terms(
    drift_func_rough, drift_func_smooth, diff_coeff_rough
):
    def drift_func(x, θ):
        return snp.concatenate((drift_func_rough(x, θ), drift_func_smooth(x, θ)))

    def outer(a, b=None):
        b = a if b is None else b
        return a.reshape(a.shape + (1,) * b.ndim) * b.reshape((1,) * a.ndim + b.shape)

    mean_and_covariance = local_gaussian_mean_and_covariance(
        drift_func_rough, drift_func_smooth, diff_coeff_rough
    )

    def hermite_polynomial_terms(t, x, y, θ):
        _, Σ_1 = mean_and_covariance(x, θ, 1)
        dim_r = drift_func_rough(x, θ).shape[0]
        x_r, x_s = x[:dim_r], x[dim_r:]
        y_r, y_s = y[:dim_r], y[dim_r:]
        Σ_1_h_1 = snp.concatenate(
            (
                (y_r - x_r - drift_func_rough(x, θ) * t) / snp.sqrt(t),
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
        Σ_1_inv = snp.array(sympy.Matrix(Σ_1).inverse_CH())
        h_1 = Σ_1_inv @ Σ_1_h_1
        outer_h_1_h_1 = outer(h_1, h_1)
        outer_h_1_h_1_h_1 = outer(h_1, outer_h_1_h_1)
        h_2 = outer_h_1_h_1 - Σ_1_inv
        outer_h_1_Σ_1_inv = outer(h_1, Σ_1_inv)
        h_3 = (
            outer_h_1_h_1_h_1
            - outer_h_1_Σ_1_inv
            - outer_h_1_Σ_1_inv.transpose((1, 0, 2))
            - outer_h_1_Σ_1_inv.transpose((2, 1, 0))
        )
        return h_1, h_2, h_3

    def Φ_1(t, x, y, θ):
        dim_r = drift_func_rough(x, θ).shape[0]
        v_r = [drift_func_rough] + [
            subscript_k(diff_coeff_rough, k) for k in range(dim_r)
        ]
        _, _, h_3 = hermite_polynomial_terms(t, x, y, θ)
        return (snp.sqrt(t) / 2) * (
            sum(
                outer(
                    outer(
                        v_hat_k(drift_func, diff_coeff_rough, k_1, dim_r)(v_r[k_2])(
                            x, θ
                        ),
                        v_r[k_1](x, θ),
                    ),
                    v_r[k_2](x, θ),
                )
                for k_1 in range(1, dim_r + 1)
                for k_2 in range(1, dim_r + 1)
            )
            * h_3[:dim_r, :dim_r, :dim_r]
        ).sum()

    def Φ_2(t, x, y, θ):
        dim_r = drift_func_rough(x, θ).shape[0]
        v_r = [drift_func_rough] + [
            subscript_k(diff_coeff_rough, k) for k in range(dim_r)
        ]
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
                outer(v_hat_k(drift_func, diff_coeff_rough, k_1, dim_r)(v_r[k_2])(x, θ))
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
                    for k in range(1, dim_r + 1)
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
        _, h_2, _ = hermite_polynomial_terms(t, x, y, θ)
        return (
            (M_rr * h_2[:dim_r, :dim_r]).sum()
            + (M_rs * h_2[:dim_r, dim_r:]).sum()
            + (M_ss * h_2[dim_r:, dim_r:]).sum()
        )

    def Φ_3(t, x, y, θ):
        dim_r = drift_func_rough(x, θ).shape[0]
        h_1, _, _ = hermite_polynomial_terms(t, x, y, θ)
        return (snp.sqrt(t**3) / 2) * (
            v_hat_k(drift_func, diff_coeff_rough, 0, dim_r)(drift_func_rough)(x, θ)
            * h_1[:dim_r]
        ).sum()

    return Φ_1, Φ_2, Φ_3


# This is for the correction term Φ_2 with the improved drift terms
def variance_correction_terms_with_improved_drift(
    drift_func_rough, drift_func_smooth, diff_coeff_rough
):
    def drift_func(x, θ):
        return snp.concatenate((drift_func_rough(x, θ), drift_func_smooth(x, θ)))

    def outer(a, b=None):
        b = a if b is None else b
        return a.reshape(a.shape + (1,) * b.ndim) * b.reshape((1,) * a.ndim + b.shape)

    mean_and_covariance = local_gaussian_improved_mean_and_covariance(
        drift_func_rough, drift_func_smooth, diff_coeff_rough
    )

    def hermite_second_order_polynomial(t, x, y, θ):
        _, Σ_1 = mean_and_covariance(x, θ, 1)
        μ_t, _ = mean_and_covariance(x, θ, t)
        dim_r = drift_func_rough(x, θ).shape[0]
        y_r, y_s = y[:dim_r], y[dim_r:]
        μ_t_r, μ_t_s = μ_t[:dim_r], μ_t[dim_r:]
        Σ_1_h_1 = snp.concatenate(
            (
                (y_r - μ_t_r) / snp.sqrt(t),
                (y_s - μ_t_s) / snp.sqrt(t) ** 3,
            )
        )
        Σ_1_inv = snp.array(sympy.Matrix(Σ_1).inverse_CH())
        h_1 = Σ_1_inv @ Σ_1_h_1
        outer_h_1_h_1 = outer(h_1, h_1)
        h_2 = outer_h_1_h_1 - Σ_1_inv
        return h_2

    def Φ_2(t, x, y, θ):
        dim_r = drift_func_rough(x, θ).shape[0]
        v_r = [drift_func_rough] + [
            subscript_k(diff_coeff_rough, k) for k in range(dim_r)
        ]
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
                outer(v_hat_k(drift_func, diff_coeff_rough, k_1, dim_r)(v_r[k_2])(x, θ))
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
                    for k in range(1, dim_r + 1)
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
        h_2 = hermite_second_order_polynomial(t, x, y, θ)
        return (
            (M_rr * h_2[:dim_r, :dim_r]).sum()
            + (M_rs * h_2[:dim_r, dim_r:]).sum()
            + (M_ss * h_2[dim_r:, dim_r:]).sum()
        )

    return Φ_2


# This returns log of LG density for Hypo-I model
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


# This returns log of LG density for Hypo-II model
def local_gaussian_log_transition_density_II(
    drift_func_smooth_1, drift_func_smooth_2, drift_func_rough, diff_coeff_rough
):

    mean_and_covariance = local_gaussian_mean_and_covariance_II(
        drift_func_smooth_1, drift_func_smooth_2, drift_func_rough, diff_coeff_rough
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


# This is the log density of LG scheme with improved drift approximation
def local_gaussian_log_transition_density_improved_drift(
    drift_func_rough, drift_func_smooth, diff_coeff_rough
):

    mean_and_covariance = local_gaussian_improved_mean_and_covariance(
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


def improved_scheme_transition_density_1(
    drift_func_rough, drift_func_smooth, diff_coeff_rough
):
    lg_log_transition_density = local_gaussian_log_transition_density(
        drift_func_rough, drift_func_smooth, diff_coeff_rough
    )

    Φ_1, Φ_2, Φ_3 = improved_scheme_correction_terms(
        drift_func_rough, drift_func_smooth, diff_coeff_rough
    )

    def transition_density(x_t, x_0, θ, t):
        Φ = Φ_1(t, x_0, x_t, θ) + Φ_2(t, x_0, x_t, θ) + Φ_3(t, x_0, x_t, θ)
        return snp.exp(lg_log_transition_density(x_t, x_0, θ, t)) * (1 + Φ)

    return transition_density


def improved_scheme_log_transition_density_2(
    drift_func_rough, drift_func_smooth, diff_coeff_rough
):
    lg_log_transition_density = local_gaussian_log_transition_density(
        drift_func_rough, drift_func_smooth, diff_coeff_rough
    )

    Φ_1, Φ_2, Φ_3 = improved_scheme_correction_terms(
        drift_func_rough, drift_func_smooth, diff_coeff_rough
    )

    def log1p_taylor_expansion(z, n=6):
        return sum((-1) ** (i + 1) * z**i / i for i in range(1, n + 1))

    def log_transition_density(x_t, x_0, θ, t):
        Φ = Φ_1(t, x_0, x_t, θ) + Φ_2(t, x_0, x_t, θ) + Φ_3(t, x_0, x_t, θ)
        return lg_log_transition_density(x_t, x_0, θ, t) + log1p_taylor_expansion(Φ)

    return log_transition_density


def improved_scheme_log_transition_density_proxy(
    drift_func_rough, drift_func_smooth, diff_coeff_rough
):
    lg_log_transition_density = local_gaussian_log_transition_density(
        drift_func_rough, drift_func_smooth, diff_coeff_rough
    )

    _, Φ_2, _ = improved_scheme_correction_terms(
        drift_func_rough, drift_func_smooth, diff_coeff_rough
    )

    def log_transition_density(x_t, x_0, θ, t):
        return lg_log_transition_density(x_t, x_0, θ, t) + Φ_2(t, x_0, x_t, θ)

    return log_transition_density


## This is the contrast function achieving the CLT with Δ = o (n^{-1/4})
def improved_scheme2_log_transition_density_proxy(
    drift_func_rough, drift_func_smooth, diff_coeff_rough
):
    lg_log_transition_density = local_gaussian_log_transition_density_improved_drift(
        drift_func_rough, drift_func_smooth, diff_coeff_rough
    )

    Φ_2 = variance_correction_terms_with_improved_drift(
        drift_func_rough, drift_func_smooth, diff_coeff_rough
    )

    def log_transition_density(x_t, x_0, θ, t):
        return lg_log_transition_density(x_t, x_0, θ, t) + Φ_2(t, x_0, x_t, θ)

    return log_transition_density
