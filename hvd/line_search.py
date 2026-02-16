import numpy as np


def backtracking_line_search(R: np.ndarray, phi_func: callable, max_step_size: float = 1) -> float:
    """backtracking line search with Armijo's condition"""
    # TODO: make this a standard procedure for all optimizers
    # TODO: implement curvature condition or use scipy's line search function
    c1 = 1e-5
    # TODO: check if this is needed
    # if step sizes are near zero, then no need to do the line search
    # if np.any(np.isclose(np.median(step[:, : self.dim_p]), np.finfo(np.double).resolution)):
    # return 1

    step_size = max_step_size
    phi = [np.linalg.norm(R)]
    s = [0, step_size]
    for _ in range(6):
        phi.append(phi_func(s[-1]))
        # when R norm is close to machine precision, it makes no sense to perform the line search
        success = phi[-1] <= (1 - c1 * s[-1]) * phi[0] or np.isclose(phi[0], np.finfo(float).eps)
        if success:
            break
        else:
            if 11 < 2:
                # cubic interpolation to compute the next step length
                d1 = -phi[-2] - phi[-1] - 3 * (phi[-2] - phi[-1]) / (s[-2] - s[-1])
                d2 = np.sign(s[-1] - s[-2]) * np.sqrt(d1**2 - phi[-2] * phi[-1])
                s_ = s[-1] - (s[-1] - s[-2]) * (-phi[-1] + d2 - d1) / (-phi[-1] + phi[-2] + 2 * d2)
                s_ = s[-1] * 0.5 if np.isnan(s_) else np.clip(s_, 0.4 * s[-1], 0.6 * s[-1])
                s.append(s_)
            else:
                s.append(s[-1] * 0.5)
    # else:
    # self.logger.warn("backtracking line search failed")
    step_size = s[-1]
    return step_size
