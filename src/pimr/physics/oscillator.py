"""
Driven damped harmonic oscillator ODE: 

x'' = -omega^2 x - 2*drag*omega*x' + F(t).

Reference integration with scipy's solve_ivp plus hand-rolled forward Euler, backward Euler and RK2 propagators in matrix form (the building blocks for embedding the dynamics in a physics-informed model).
"""

import numpy as np
from scipy.integrate import solve_ivp


def rhs(t, state, omega: float = .573, drag: float = 0.0, t_force=None, force=None):
    """Right-hand side of the driven damped system [x, v]' = [v, -omega^2 x - 2*drag*omega*v + F(t)].

    The driving force is supplied as samples `force` on the time grid `t_force`;
    solve_ivp evaluates at arbitrary internal times, so F(t) is obtained by
    linear interpolation between the samples.
    """
    x, v = state
    F = 0.0 if force is None else np.interp(t, t_force, force)
    return [v, -omega**2 * x - 2.0 * drag * omega * v + F]


def SHO_func(t, x, v, omega):
    return [v, -omega**2 * x]


def simulate(N_POINTS: int = 100, T_END: float = 10.0, omega: float = 0.573, drag: float = 0.0, force=None, init_cond=[1.0, 0.0]):
    """Integrate the driven damped oscillator and return (t, x_numeric, x_analytic).

    Solves x'' + 2*drag*omega*x' + omega^2 x = F(t). `force` is an array of
    length N_POINTS holding the driving force at each point of the time grid
    (linearly interpolated in between); None means unforced. x_analytic is the
    exact solution of the free underdamped oscillator (F=0, drag < 1) — there
    is no closed form for an arbitrary force, so only compare against it when
    the force is zero.
    """
    X0, V0 = init_cond
    t_eval = np.linspace(0.0, T_END, N_POINTS)

    if force is None:
        force = np.zeros(N_POINTS)
    force = np.asarray(force, dtype=float)
    if force.shape != t_eval.shape:
        raise ValueError(f"force must be a 1-D array of length N_POINTS={N_POINTS}, got shape {force.shape}")

    sol = solve_ivp(
        rhs,
        t_span=(0.0, T_END),
        y0=init_cond,
        t_eval=t_eval,
        args=(omega, drag, t_eval, force),
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )

    omega_d = omega * np.sqrt(1.0 - drag**2) if drag < 1.0 else np.nan
    x_analytic = np.exp(-drag * omega * t_eval) * (
        X0 * np.cos(omega_d * t_eval)
        + ((V0 + drag * omega * X0) / omega_d) * np.sin(omega_d * t_eval)
    )
    return sol.t, sol.y[0], x_analytic


def step_matrix(omega: float = 0.573, drag: float = 0.0):
    return np.array([[0, 1], [-omega ** 2, -2 * drag * omega]], dtype=np.float32)


def FE(dt: float = 0.01, N_POINTS: int = 100, t_start: float = 0.0, omega: float = 0.573, drag: float = 0.0, init_cond=[1.0, 0.0], force=None):
    """
    Solve the oscillator with the explicit (forward) Euler method in matrix form. The system [x, v]' = A @ [x, v]
    Parameters
    ----------
    dt : float, optional
        Step size of the time grid.
    N_POINTS : int, optional
        Number of time steps (including t=t_start).
    t_start : float, optional
        Simulation starting point.
    init_cond : list of float, optional
        Initial state [x0, v0] (position and velocity).
    force : ndarray of length N_POINTS, optional
        Driving force at each grid point; None means unforced.
    Returns
    -------
    t : ndarray
        Time grid.
    state : ndarray, shape (n_points, 2)
        State vector [:][x,v] at each time step.
    """
    if force is None:
        force = np.zeros(N_POINTS)
    if len(force) != N_POINTS:
        raise ValueError(f"force must be a 1-D array of length N_POINTS={N_POINTS}, got shape {np.asarray(force, dtype=float).shape}")

    A_mat = step_matrix(omega=omega, drag=drag)

    t_array = t_start + dt * np.arange(0, N_POINTS, dtype=np.float32)
    state = np.empty((N_POINTS, 2))

    state[0] = np.array(init_cond)

    M = np.eye(2) + dt * A_mat  # propagator is constant, build it once
    for i in range(N_POINTS - 1):
        state[i + 1] = M @ state[i] + dt * np.array([0.0, force[i]])

    return t_array, state


def BE(dt: float = 0.01, N_POINTS: int = 100, t_start: float = 0.0, omega: float = 0.573, drag: float = 0.0, init_cond=[1.0, 0.0], force=None):
    """
    Solve the oscillator with the implicit (backward) Euler method in matrix form. Backward Euler evaluates the RHS at the *new* state: state[i+1] = state[i] + dt * (A @ state[i+1] + b[i+1]) with b = [0, F(t)]. Solving for state[i+1] gives state[i+1] = (I - dt*A)^-1 @ (state[i] + dt * b[i+1]).
    Parameters
    ----------
    dt : float, optional
        Step size of the time grid.
    N_POINTS : int, optional
        Number of time steps (including t=t_start).
    t_start : float, optional
        Simulation starting point.
    init_cond : list of float, optional
        Initial state [x0, v0] (position and velocity).
    force : ndarray of length N_POINTS, optional
        Driving force at each grid point; None means unforced.
    Returns
    -------
    t : ndarray
        Time grid.
    state : ndarray, shape (n_points, 2)
        State vector [:][x,v] at each time step.
    """
    if force is None:
        force = np.zeros(N_POINTS)
    if len(force) != N_POINTS:
        raise ValueError(f"force must be a 1-D array of length N_POINTS={N_POINTS}, got shape {np.asarray(force, dtype=float).shape}")

    A_mat = step_matrix(omega=omega, drag=drag)

    t_array = t_start + dt * np.arange(0, N_POINTS, dtype=np.float32)
    state = np.empty((N_POINTS, 2))

    state[0] = np.array(init_cond)

    M = np.linalg.inv(np.eye(2) - dt * A_mat)  # propagator is constant, invert once
    for i in range(N_POINTS - 1):
        state[i + 1] = M @ (state[i] + dt * np.array([0.0, force[i + 1]]))

    return t_array, state


def forward_euler_vectorized(dt: float = 0.01, N_POINTS: int = 100, t_start: float = 0.0, omega: float = 0.573, drag: float = 0.0, init_cond=[1.0, 0.0]):
    """
    Forward Euler without the Python loop, via eigendecomposition.
    The recurrence state[i+1] = M @ state[i] with constant M = I + dt*A has the closed form state[i] = M^i @ state[0]. Diagonalising M = P diag(evals) P^-1
    turns the matrix powers into element-wise powers of the eigenvalues, which numpy evaluates for all i at once. Same trajectory as FE up to round-off.
    Only the undriven (force-free) oscillator is supported.
    Parameters
    ----------
    dt : float, optional
        Step size of the time grid.
    N_POINTS : int, optional
        Number of time steps (including t=t_start).
    t_start : float, optional
        Simulation starting point.
    init_cond : list of float, optional
        Initial state [x0, v0] (position and velocity).
    Returns
    -------
    t : ndarray
        Time grid.
    state : ndarray, shape (n_points, 2)
        State at each time step: state[:, 0] is position x, state[:, 1] is velocity v.
    """
    t_array = t_start + dt * np.arange(0, N_POINTS, dtype=np.float32)

    M = np.eye(2) + dt * step_matrix(omega=omega, drag=drag)
    evals, P = np.linalg.eig(M)
    c = np.linalg.solve(P, np.array(init_cond, dtype=complex))
    powers = evals[None, :] ** np.arange(N_POINTS)[:, None]   # (n_points, 2)
    state = (powers * c) @ P.T   # row i equals P @ (evals**i * c)

    return t_array, state.real


def RK2(dt: float = 0.01, N_POINTS: int = 100, t_start: float = 0.0, omega: float = 0.573, drag: float = 0.0, init_cond=[1.0, 0.0], force=None):
    """
    Solve the oscillator with second order Runge-Kutta (Heun) method in matrix form.
    For the affine system y' = A @ y + b(t) with b = [0, F(t)], the two stages
    k1 = A y_i + b_i and k2 = A (y_i + dt*k1) + b_{i+1} combine into
    y_{i+1} = (I + dt*A + dt^2/2 * A^2) y_i + dt/2 * (b_i + b_{i+1}) + dt^2/2 * A @ b_i,
    i.e. the same constant propagator as before plus a trapezoid average of the
    force and a cross-term carrying the stage-1 force through stage 2.
    Parameters
    ----------
    dt : float, optional
        Step size of the time grid.
    N_POINTS : int, optional
        Number of time steps (including t=t_start).
    t_start : float, optional
        Simulation starting point.
    init_cond : list of float, optional
        Initial state [x0, v0] (position and velocity).
    force : ndarray of length N_POINTS, optional
        Driving force at each grid point; None means unforced.
    Returns
    -------
    t : ndarray
        Time grid.
    state : ndarray, shape (n_points, 2)
        State vector [:][x,v] at each time step.
    """
    if force is None:
        force = np.zeros(N_POINTS)
    if len(force) != N_POINTS:
        raise ValueError(f"force must be a 1-D array of length N_POINTS={N_POINTS}, got shape {np.asarray(force, dtype=float).shape}")

    A_mat = step_matrix(omega=omega, drag=drag)

    t_array = t_start + dt * np.arange(0, N_POINTS, dtype=np.float32)
    state = np.empty((N_POINTS, 2))

    state[0] = np.array(init_cond)

    M = np.eye(2) + dt * A_mat + (dt ** 2 / 2) * (A_mat @ A_mat)
    for i in range(N_POINTS - 1):
        b_i = np.array([0.0, force[i]])
        b_next = np.array([0.0, force[i + 1]])
        state[i + 1] = M @ state[i] + (dt / 2) * (b_i + b_next) + (dt**2 / 2) * (A_mat @ b_i)

    return t_array, state
