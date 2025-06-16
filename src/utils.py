"""
Utilities for visualising the interior‑point method used in HW4.

The module provides three public helpers:

1. plot_feasible_and_path(...)
   Draws the feasible region (polygon in 2‑D or triangle/simplex in 3‑D)
   together with the *outer*‑iteration points stored in the optimiser’s
   history.

2. plot_objective_vs_outer(history, **kwargs)
   A simple line plot of the objective value returned at every outer
   iteration.

3. print_final_candidate(...)
   Prints the objective value as well as inequality / equality residuals
   of the final point so you can copy‑paste it into the PDF report.

All functions only rely on NumPy and Matplotlib, so there are no extra
dependencies.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – enables 3‑D projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_feasible(
    x: np.ndarray,
    ineq_constraints: Sequence[Callable[[np.ndarray], Tuple[float, np.ndarray, Optional[np.ndarray]]]],
    A: Optional[np.ndarray],
    b: Optional[np.ndarray],
    tol: float = 1e-9,
) -> bool:
    """Return **True** if *x* satisfies all constraints up to *tol*."""
    # Inequalities g_i(x) ≤ 0
    for g_fun in ineq_constraints:
        g_val, _, _ = g_fun(x)
        if g_val > tol:
            return False

    # Equalities A x = b
    if A is not None and A.size > 0:
        if not np.allclose(A @ x, b, atol=tol):
            return False

    return True


def _grid_for_2d(xlim: Tuple[float, float], ylim: Tuple[float, float], points: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """Create a dense mesh‑grid that covers *xlim* × *ylim*."""
    xs = np.linspace(*xlim, points)
    ys = np.linspace(*ylim, points)
    return np.meshgrid(xs, ys, indexing="xy")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_feasible_and_path(
    ineq_constraints: Sequence[Callable[[np.ndarray], Tuple[float, np.ndarray, Optional[np.ndarray]]]],
    eq_constraints_mat: Optional[np.ndarray],
    eq_constraints_rhs: Optional[np.ndarray],
    history: Sequence,  # Sequence[HistoryEntry] – imported lazily to avoid circular import
    *,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    zlim: Optional[Tuple[float, float]] = None,
    title: str | None = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    resolution: int = 400,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualise the feasible region (grey) and the central‑path (red) defined
    by *history*.

    Parameters
    ----------
    ineq_constraints
        List of callables returning (value, gradient, hessian) just like in the
        assignment.
    eq_constraints_mat, eq_constraints_rhs
        Matrices defining Ax = b (may be `None` / empty for the LP example).
    history
        The optimiser’s *outer* iteration history (see `constrainedMinimizer`).
    xlim, ylim, zlim
        Plot window limits.  If *None*, they are inferred from the history.
    resolution
        Grid density for the 2‑D feasible‑region fill.
    """
    # ---------------------------------------------------------------------
    # Gather path information
    # ---------------------------------------------------------------------
    path = np.array([h.x for h in history])
    dim = path.shape[1]

    # Figure / axes handling
    if fig is None or ax is None:
        if dim == 3:
            fig = plt.figure(figsize=(8, 6)) if fig is None else fig
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(figsize=(6, 6))

    # ---------------------------------------------------------------------
    # 2‑D case (LP example)
    # ---------------------------------------------------------------------
    if dim == 2:
        # Auto limits if not provided
        if xlim is None:
            pad = 0.25
            xlim = (path[:, 0].min() - pad, path[:, 0].max() + pad)
        if ylim is None:
            pad = 0.25
            ylim = (path[:, 1].min() - pad, path[:, 1].max() + pad)

        # Evaluate feasibility on a dense grid
        Xg, Yg = _grid_for_2d(xlim, ylim, resolution)
        pts = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
        feas = np.array(
            [
                _is_feasible(p, ineq_constraints, eq_constraints_mat, eq_constraints_rhs)
                for p in pts
            ],
            dtype=bool,
        ).reshape(Xg.shape)

        # Colour‑fill feasible region
        ax.contourf(
            Xg,
            Yg,
            feas.astype(float),
            levels=[0.5, 1.5],
            colors=["#e0e0e0"],
            alpha=0.5,
        )

        # Draw central path
        ax.plot(
            path[:, 0],
            path[:, 1],
            marker="o",
            color="tab:red",
            linewidth=1.5,
            label="central path (outer iters)",
        )

        ax.scatter(
            path[-1, 0],
            path[-1, 1],
            color="tab:green",
            s=80,
            zorder=5,
            label="final point",
        )

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")

    # ---------------------------------------------------------------------
    # 3‑D case (QP example: simplex in ℝ³)
    # ---------------------------------------------------------------------
    elif dim == 3:
        # Try to guess / construct vertices of the simplex defined by
        #   A x = b,    x ≥ 0.
        verts: List[np.ndarray] = []
        if (
            eq_constraints_mat is not None
            and eq_constraints_mat.shape == (1, 3)
            and eq_constraints_rhs is not None
            and len(ineq_constraints) >= 3
        ):
            a = eq_constraints_mat[0].astype(float)
            b_val = float(eq_constraints_rhs[0])

            # Intersections with coordinate axes (set other variables to 0)
            for i in range(3):
                if abs(a[i]) > 1e-12:
                    v = np.zeros(3)
                    v[i] = b_val / a[i]
                    # Feasibility check (x_i must be non‑negative etc.)
                    if _is_feasible(v, ineq_constraints, eq_constraints_mat, eq_constraints_rhs) and v[i] >= 0:
                        verts.append(v)

        # If we managed to identify 3 vertices, draw the triangle
        if len(verts) == 3:
            tri = Poly3DCollection([verts], alpha=0.3, facecolor="#e0e0e0", edgecolor="k")
            ax.add_collection3d(tri)

            # Also draw edges for clarity
            for i in range(3):
                v1, v2 = verts[i], verts[(i + 1) % 3]
                ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], color="k", linewidth=1)

        # Plot central path
        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            marker="o",
            color="tab:red",
            linewidth=1.5,
            label="central path (outer iters)",
        )
        ax.scatter(
            path[-1, 0],
            path[-1, 1],
            path[-1, 2],
            color="tab:green",
            s=80,
            zorder=5,
            label="final point",
        )

        # Axes limits
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if zlim is not None:
            ax.set_zlim(*zlim)

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$z$")
        ax.view_init(elev=20, azim=45)

    else:
        raise ValueError("Only 2‑D or 3‑D problems are supported by these plotting utilities.")

    # House‑keeping
    if title is not None:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_objective_vs_outer(
    history: Sequence,
    *,
    title: str | None = "Objective value vs. outer iteration",
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Line‑plot of *f*‑values stored in *history*."""
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    ks = [h.k for h in history]
    fs = [h.f for h in history]

    ax.plot(ks, fs, marker="o", linewidth=1.5, color="tab:blue")
    ax.set_xlabel("outer iteration $k$")
    ax.set_ylabel(r"$f(x_k)$")
    if title is not None:
        ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    return fig, ax


def print_final_candidate(
    func: Callable[[np.ndarray], Tuple[float, np.ndarray, Optional[np.ndarray]]],
    x_star: np.ndarray,
    ineq_constraints: Sequence[Callable[[np.ndarray], Tuple[float, np.ndarray, Optional[np.ndarray]]]],
    eq_constraints_mat: Optional[np.ndarray],
    eq_constraints_rhs: Optional[np.ndarray],
    *,
    label: str | None = None,
    tol: float = 1e-8,
) -> None:
    """
    Pretty‑print objective and constraint values at *x_star*.

    Example output
    --------------
    >>> Final candidate (LP example)
        f(x) = 42.000000
        g_1(x) = -0.000001
        ...
        A x - b = [1.2e-09]
    """
    if label is None:
        label = "Final candidate"

    print(f"\n>>> {label}")
    f_val, *_ = func(x_star)
    print(f"    f(x) = {f_val:.6f}")

    for idx, g_fun in enumerate(ineq_constraints, start=1):
        g_val, *_ = g_fun(x_star)
        sign = "+" if g_val > 0 else ""
        print(f"    g_{idx}(x) = {sign}{g_val:.6e}")

    if eq_constraints_mat is not None and eq_constraints_mat.size > 0:
        resid = eq_constraints_mat @ x_star - eq_constraints_rhs
        print(f"    A x - b = {resid}")

    feas = _is_feasible(x_star, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, tol=tol)
    print(f"    Feasible (within tol={tol})? {feas}\n")