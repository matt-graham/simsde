import symnum.numpy as snp
from simsde.operators import v_hat_k, subscript_k, square_bracket


def euler_maruyama_step(drift_func, diff_coeff):
    def step_func(x, θ, n, t):
        return x + t * drift_func(x, θ) + t**0.5 * diff_coeff(x, θ) @ n

    return step_func


def elliptic_weak_order_2_step(drift_func, diff_coeff):
    def step_func(x, θ, n, t):
        dim_x = x.shape[0]
        v = [drift_func] + [subscript_k(diff_coeff, k) for k in range(dim_x)]
        b = [t] + [t**0.5 * n[k] for k in range(dim_x)]
        b_tilde = [None] * 2 + [t**0.5 * n[k] for k in range(dim_x, 2 * dim_x - 1)]
        return (
            x
            + sum(v[k](x, θ) * b[k] for k in range(dim_x + 1))
            + sum(
                v_hat_k(drift_func, diff_coeff, k_1)(v[k_2])(x, θ)
                * (b[k_1] * b[k_2] - (t if (k_1 == k_2 and k_1 != 0) else 0))
                for k_1 in range(dim_x + 1)
                for k_2 in range(dim_x + 1)
            )
            / 2
            + sum(
                square_bracket(drift_func, diff_coeff, k_1, k_2)(x, θ)
                * (b[k_1] * b_tilde[k_2])
                for k_1 in range(1, dim_x + 1)
                for k_2 in range(k_1 + 1, dim_x + 1)
            )
            / 2
        )

    return step_func


def hypoelliptic_weak_order_2_step(
    drift_func_rough, drift_func_smooth, diff_coeff_rough
):
    def drift_func(x, θ):
        return snp.concatenate((drift_func_rough(x, θ), drift_func_smooth(x, θ)))

    def step_func(x, θ, n, t):
        dim_r = drift_func_rough(x, θ).shape[0]
        x_r, x_s = x[:dim_r], x[dim_r:]
        v_r = [drift_func_rough] + [
            subscript_k(diff_coeff_rough, k) for k in range(dim_r)
        ]
        b = [t] + [t**0.5 * n[k] for k in range(dim_r)]
        b_hat = [None] + [t**0.5 * n[k] for k in range(dim_r, 2 * dim_r)]
        b_tilde = [None] + [t**0.5 * n[k] for k in range(2 * dim_r, 3 * dim_r)]
        w = [None] * 2 + [t**0.5 * n[k] for k in range(3 * dim_r, 4 * dim_r - 1)]

        def ζ(k_1, k_2):
            if k_1 == k_2 == 0:
                return t**2 / 2
            elif k_1 == 0:
                return (t / 2) * (b[k_2] + b_hat[k_2] / snp.sqrt(3))
            elif k_2 == 0:
                return b[k_1] * t - ζ(0, k_1)
            elif k_1 == k_2:
                return (b[k_1] ** 2 - t) / 2
            elif k_1 < k_2:
                return (b[k_1] * b[k_2] + b[k_1] * b_tilde[k_2]) / 2
            else:
                return (b[k_1] * b[k_2] - b[k_2] * b_tilde[k_1]) / 2

        def η(k_1, k_2):
            if k_1 == k_2 == 0:
                # This value should never be used
                return None
            elif k_2 == 0:
                return ζ(k_1, 0) * t / 2 - b[k_1] * t**2 / 12
            elif k_1 == 0:
                return ζ(k_2, 0) * t - b[k_2] * t**2 / 3
            elif k_1 == k_2:
                return ζ(k_1, k_2) * t / 3 - (b_tilde[k_1] ** 2 - t) / (6 * snp.sqrt(2))
            elif k_1 < k_2:
                return ζ(k_1, k_2) * t / 3 - (
                    b_tilde[k_1] * b_tilde[k_2] - b_tilde[k_1] * w[k_2]
                ) / (6 * snp.sqrt(2))
            else:
                return ζ(k_1, k_2) * t / 3 - (
                    b_tilde[k_1] * b_tilde[k_2] - b_tilde[k_2] * w[k_1]
                ) / (6 * snp.sqrt(2))

        return snp.concatenate(
            (
                x_r
                + sum(v_r[k](x, θ) * b[k] for k in range(dim_r + 1))
                + sum(
                    v_hat_k(drift_func, diff_coeff_rough, k_1, dim_r)(v_r[k_2])(x, θ)
                    * ζ(k_1, k_2)
                    for k_1 in range(dim_r + 1)
                    for k_2 in range(dim_r + 1)
                )
                / 2,
                x_s
                + drift_func_smooth(x, θ) * t
                + sum(
                    v_hat_k(drift_func, diff_coeff_rough, k, dim_r)(drift_func_smooth)(
                        x, θ
                    )
                    * ζ(k, 0)
                    for k in range(dim_r + 1)
                )
                + sum(
                    v_hat_k(drift_func, diff_coeff_rough, k_1, dim_r)(
                        v_hat_k(drift_func, diff_coeff_rough, k_2, dim_r)(
                            drift_func_smooth
                        )
                    )(x, θ)
                    * η(k_1, k_2)
                    for k_1 in range(dim_r + 1)
                    for k_2 in range(dim_r + 1)
                    if not (k_1 == k_2 == 0)
                ),
            )
        )

    return step_func


def hypoelliptic_local_gaussian_step(
    drift_func_rough, drift_func_smooth, diff_coeff_rough
):
    def drift_func(x, θ):
        return snp.concatenate((drift_func_rough(x, θ), drift_func_smooth(x, θ)))

    def step_func(x, θ, n, t):
        dim_r = drift_func_rough(x, θ).shape[0]
        x_r, x_s = x[:dim_r], x[dim_r:]
        b = snp.sqrt(t) * n[:dim_r]
        b_tilde = snp.sqrt(t**3) * (n[:dim_r] * n[dim_r:] / snp.sqrt(3)) / 2
        return snp.concatenate(
            [
                x_r + drift_func_rough(x, θ) * t + diff_coeff_rough(x, θ) @ b,
                x_s
                + drift_func_smooth(x, θ) * t
                + v_hat_k(drift_func, diff_coeff_rough, 0, dim_r)(drift_func_smooth)(
                    x, θ
                )
                * t**2
                / 2
                + sum(
                    v_hat_k(drift_func, diff_coeff_rough, k, dim_r)(drift_func_smooth)(
                        x, θ
                    )
                    * b_tilde[k - 1]
                    for k in range(1, dim_r + 1)
                ),
            ]
        )

    return step_func


# This is the sampling scheme for the model Hypo-II
# smooth_1 correponds to the most smooth part of the system, for example,
# smooth_1 = position, smooth_2 = momentum in the generalized Langevin equation
def hypoelliptic_II_local_gaussian_step(
    drift_func,
    drift_func_rough,
    drift_func_smooth_1,
    drift_func_smooth_2,
    diff_coeff_rough,
):
    def step_func(x, θ, n, t):
        dim_r = drift_func_rough(x, θ).shape[0]
        dim_s2 = drift_func_smooth_2(x, θ).shape[0]
        x_r, x_s_2, x_s_1 = x[:dim_r], x[dim_r : dim_r + dim_s2], x[dim_r + dim_s2 :]
        b = snp.sqrt(t) * n[:dim_r]
        int_bm_time = (
            snp.sqrt(t**3) * (n[:dim_r] + n[dim_r : 2 * dim_r] / snp.sqrt(3)) / 2
        )
        int_bm_time_time = (
            snp.sqrt(t**5)
            * (
                n[:dim_r]
                + snp.sqrt(3) * n[dim_r : 2 * dim_r] / 2
                + n[2 * dim_r :] / snp.sqrt(20)
            )
            / 6
        )
        return snp.concatenate(
            [
                # integrator for the rough component
                x_r + drift_func_rough(x, θ) * t + diff_coeff_rough(x, θ) @ b,
                # integrator for the second smooth component
                x_s_2
                + drift_func_smooth_2(x, θ) * t
                + v_hat_k(drift_func, diff_coeff_rough, 0, dim_r)(drift_func_smooth_2)(
                    x, θ
                )
                * t**2
                / 2
                + sum(
                    v_hat_k(drift_func, diff_coeff_rough, k, dim_r)(
                        drift_func_smooth_2
                    )(x, θ)
                    * int_bm_time[k - 1]
                    for k in range(1, dim_r + 1)
                ),
                # integrator for the most smooth component
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
                / 6
                + sum(
                    v_hat_k(drift_func, diff_coeff_rough, k, dim_r)(
                        v_hat_k(drift_func, diff_coeff_rough, 0, dim_r)(
                            drift_func_smooth_1
                        )
                    )(x, θ)
                    * int_bm_time_time[k - 1]
                    for k in range(1, dim_r + 1)
                ),
            ]
        )

    return step_func
