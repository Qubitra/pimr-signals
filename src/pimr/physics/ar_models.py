"""Autoregressive model fitting and oscillatory-mode analysis."""

import numpy as np

def AR2(x):
    x = np.asarray(x, dtype=float)

    coef_mat = np.column_stack((
        np.ones(len(x) - 2),
        x[1:-1],
        x[:-2],
    ))
    target = x[2:]

    return np.linalg.lstsq(coef_mat, target, rcond=None)[0]

def AR4(x):
    x = np.asarray(x, dtype=float)
    target = x[4:]
    coefficient = np.column_stack(
        [np.ones(len(target))] +
        [x[4 - lag : len(x) - lag] for lag in range(1, 5)]
    )
    return np.linalg.lstsq(coefficient, target, rcond=None)[0]


def oscillatory_modes(coefficients):
    # coefficients are [intercept, phi1, ..., phip]
    phi = np.asarray(coefficients[1:], dtype=float)

    # Roots of lambda**p - phi1*lambda**(p-1) - ... - phip
    roots = np.roots(np.r_[1.0, -phi])
    # stable = np.all(np.abs(roots) < 1) # No stability output for now!
    # print("   - Stable AR model:", stable)
    
    modes = []
    for root in roots:
        if root.imag > 1e-10:  # one root from each conjugate pair
            radius = abs(root)
            modes.append({
                "radius": radius,
                "omega": abs(np.angle(root)),
                "damping": -np.log(radius) if radius > 0 else np.inf,
            })
        # else:
        #     print("AR(2) has no complex-root")
            
    return modes