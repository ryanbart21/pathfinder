# --- PSO from pathfinder_6_4.py ---
def particle_swarm_6_4(f, x_lb, x_ub, n=10, inertia=0.5, self_influence=1.8, social_influence=1.8, max_vel=1.0, f_tol=1e-6):
    global nfev
    global X_history
    nfev = 0
    X_history = []

    k = 0
    lhs_raw = lhsmdu.sample(len(x_lb), n)
    lhs_samples = np.array(lhs_raw).T
    X = x_lb + (x_ub - x_lb) * lhs_samples
    dX = (np.random.rand(n, len(x_lb))*2 - 1) * max_vel     
    X_best = X.copy()
    f_vals_best = np.array([f(xi) for xi in X_best])

    n_conv = 0
    f_best_prev = np.inf
    converged = False
    while not converged:
        k += 1
        f_vals = np.array([f(xi) for xi in X])
        swarm_best_i = 0
        for i in range(n):
            if f_vals[i] < f_vals_best[i]:
                X_best[i,:] = X[i,:]
                f_vals_best[i] = f_vals[i]
            if f_vals[i] < f_vals[swarm_best_i]:
                swarm_best_i = i

        swarm_best = X[swarm_best_i,:].copy()

        for i in range(n):
            dX[i,:] = inertia*dX[i,:] + self_influence*(X_best[i,:]-X[i,:]) + social_influence*(swarm_best - X[i,:])
            dX[i,:] = np.maximum(np.minimum(dX[i,:], max_vel),-max_vel)
            X[i,:] += dX[i,:]
            X[i,:] = np.maximum(np.minimum(X[i,:], x_ub), x_lb)
            for j in range(len(x_lb)):
                if X[i,j] < x_lb[j]:
                    X[i,j] = x_lb[j]
                    dX[i,j] *= -0.5
                elif X[i,j] > x_ub[j]:
                    X[i,j] = x_ub[j]
                    dX[i,j] *= -0.5

        f_min = np.min(f_vals_best)

        if np.abs(f_best_prev - f_min) < f_tol:
            n_conv += 1
            if n_conv > 2:
                converged = True
        else:
            n_conv = 0

        f_best_prev = f_min

        X_history.append(X.copy())
        r_x = X_best[np.argmin(f_vals_best)]
        max_vel *= 0.9
    return r_x, f(r_x), nfev
import numpy as np
import math
import os
import importlib.util
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import BSpline
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import lhsmdu
import heapq

X_history = []

# ── Config ─────────────────────────────────────────────────────────────────
WORLD_SCALE = 1000.0
RRT_STEP = 0.05 * WORLD_SCALE
RRT_GOAL_RADIUS = 0.08 * WORLD_SCALE
COLLISION_SAMPLE_SPACING = 0.005 * WORLD_SCALE
LOCAL_BOUNDS_MARGIN = 0.10 * WORLD_SCALE
RHO   = 4.0                    # turning radius stays in meters
N_BOXES = 14
BOX_SEED = 23
SAVE_GIFS = False
COLLISION_CLEARANCE = 0.0
COLLISION_CONSTRAINT_SAMPLES = 260
Z_HEADROOM = 10.0
IPM_MAX_ITERS = 700
IPM_ACCEPT_MARGIN = 0.3
RUN_IPM_FROM_RRT = True
RUN_GCS = True

# Solver toggles
RUN_PSO_CURRENT = False
RUN_PSO_LEGACY = True
RUN_PSO_6_4 = False
RUN_IPM_ON_TOP_OF_PSO = True

# Plot toggles
PLOT_PSO_CURRENT = False
PLOT_PSO_LEGACY = True
PLOT_PSO_6_4 = False
PLOT_IPM_FROM_RRT = True
PLOT_PSO_IPM = True
PLOT_GCS = True

def boxes_overlap_xy(a, b, gap=0.015):
    return not (
        a[3] + gap <= b[0] or b[3] + gap <= a[0] or
        a[4] + gap <= b[1] or b[4] + gap <= a[1]
    )


def generate_base_boxes(n_boxes=N_BOXES, seed=BOX_SEED, max_attempts=25000):
    rng = np.random.default_rng(seed)
    boxes = []
    pad = 0.04

    for _ in range(max_attempts):
        width = rng.uniform(0.07, 0.14)
        depth = rng.uniform(0.07, 0.15)
        height = rng.uniform(0.06, 0.30)
        x0 = rng.uniform(pad, 1.0 - pad - width)
        y0 = rng.uniform(pad, 1.0 - pad - depth)
        candidate = (x0, y0, 0.0, x0 + width, y0 + depth, height)

        if all(not boxes_overlap_xy(candidate, existing) for existing in boxes):
            boxes.append(candidate)
            if len(boxes) == n_boxes:
                return boxes

    raise RuntimeError("Failed to generate non-overlapping boxes.")


# Box obstacles are generated in normalized map units and then scaled into meters.
# Box format: (x_min, y_min, z_min, x_max, y_max, z_max)
BASE_BOXES = generate_base_boxes()

BOXES = [tuple(value * WORLD_SCALE for value in box) for box in BASE_BOXES]
TALLEST_BOX_Z = max(box[5] for box in BOXES)
SPACE_BOUNDS = (0.0, WORLD_SCALE, 0.0, WORLD_SCALE, 0.0, TALLEST_BOX_Z + Z_HEADROOM)

START_Z = 5.0

def sample_free_point(rng, boxes, x_bounds, y_bounds, z_bounds, xy_margin=0.0, z_margin=0.0, obstacle_margin=0.0, max_attempts=5000):
    x_lo, x_hi = x_bounds
    y_lo, y_hi = y_bounds
    z_lo, z_hi = z_bounds

    for _ in range(max_attempts):
        candidate = np.array([
            rng.uniform(x_lo + xy_margin, x_hi - xy_margin),
            rng.uniform(y_lo + xy_margin, y_hi - xy_margin),
            z_lo if abs(z_hi - z_lo) < 1e-9 else rng.uniform(z_lo + z_margin, z_hi - z_margin),
        ])
        if not any(
            box[0] - obstacle_margin <= candidate[0] <= box[3] + obstacle_margin and
            box[1] - obstacle_margin <= candidate[1] <= box[4] + obstacle_margin and
            box[2] - obstacle_margin <= candidate[2] <= box[5] + obstacle_margin
            for box in boxes
        ):
            return tuple(candidate)

    raise RuntimeError("Failed to sample a free point outside the boxes.")

rng = np.random.default_rng(7)
START = sample_free_point(
    rng,
    BOXES,
    (SPACE_BOUNDS[0], SPACE_BOUNDS[1]),
    (SPACE_BOUNDS[2], SPACE_BOUNDS[3]),
    (START_Z, START_Z),
    xy_margin=0.05 * WORLD_SCALE,
    z_margin=0.0,
)


GOAL_Z = 0.5 * TALLEST_BOX_Z
GOAL = sample_free_point(
    rng,
    BOXES,
    (SPACE_BOUNDS[0], SPACE_BOUNDS[1]),
    (SPACE_BOUNDS[2], SPACE_BOUNDS[3]),
    (GOAL_Z, GOAL_Z),
    xy_margin=0.05 * WORLD_SCALE,
    z_margin=0.0,
)

while math.dist(START, GOAL) < 0.20 * WORLD_SCALE:
    GOAL = sample_free_point(
        rng,
        BOXES,
        (SPACE_BOUNDS[0], SPACE_BOUNDS[1]),
        (SPACE_BOUNDS[2], SPACE_BOUNDS[3]),
        (GOAL_Z, GOAL_Z),
        xy_margin=0.05 * WORLD_SCALE,
        z_margin=0.0,
    )

def build_bspline(control_pts, degree=3, n_samples=200):
    control_pts = np.array(control_pts)

    n = len(control_pts)
    k = min(degree, n - 1)

    # clamped knot vector
    knots = np.concatenate((
        np.zeros(k),
        np.linspace(0, 1, n - k + 1),
        np.ones(k)
    ))

    t = np.linspace(0, 1, n_samples)

    spline_x = BSpline(knots, control_pts[:,0], k)
    spline_y = BSpline(knots, control_pts[:,1], k)
    spline_z = BSpline(knots, control_pts[:,2], k)

    x = spline_x(t)
    y = spline_y(t)
    z = spline_z(t)

    dx = spline_x.derivative(1)(t)
    dy = spline_y.derivative(1)(t)
    dz = spline_z.derivative(1)(t)

    ddx = spline_x.derivative(2)(t)
    ddy = spline_y.derivative(2)(t)
    ddz = spline_z.derivative(2)(t)

    return x, y, z, dx, dy, dz, ddx, ddy, ddz

def compute_curvature(dx, dy, dz, ddx, ddy, ddz):
    v1 = np.stack([dx, dy, dz], axis=1)
    v2 = np.stack([ddx, ddy, ddz], axis=1)
    num = np.linalg.norm(np.cross(v1, v2), axis=1)
    denom = np.linalg.norm(v1, axis=1)**3 + 1e-6
    return num / denom

def sample_seg(x, y, z, th, L, kind, rho, n=80):
    pts = []
    if kind == 'S':
        ss = np.linspace(0, L, max(3, int(L/0.3)+1))
        for s in ss: pts.append((x+s*math.cos(th), y+s*math.sin(th), z))
        x += L*math.cos(th); y += L*math.sin(th)
    elif kind == 'L':
        cx = x - rho*math.sin(th); cy = y + rho*math.cos(th)
        da = L / rho
        ns = max(3, int(da*rho/0.3)+1)
        for a in np.linspace(0, da, ns):
            phi = th + a; pts.append((cx+rho*math.sin(phi), cy-rho*math.cos(phi), z))
        th += da; x, y, z = pts[-1]
    else:  # R
        cx = x + rho*math.sin(th); cy = y - rho*math.cos(th)
        da = L / rho
        ns = max(3, int(da*rho/0.3)+1)
        for a in np.linspace(0, da, ns):
            phi = th - a; pts.append((cx-rho*math.sin(phi), cy+rho*math.cos(phi), z))
        th -= da; x, y, z = pts[-1]
    return pts, x, y, th

# ── Collision helpers ───────────────────────────────────────────────────────
def pt_in_box(px, py, pz, box, mg=0.3):
    return (
        box[0] - mg <= px <= box[3] + mg and
        box[1] - mg <= py <= box[4] + mg and
        box[2] - mg <= pz <= box[5] + mg
    )

def seg_clear(ax, ay, az, bx, by, bz, boxes, mg=0.5):
    """Check a line segment is clear of all boxes."""
    n = max(20, int(math.dist((ax, ay, az), (bx, by, bz)) / COLLISION_SAMPLE_SPACING))
    for t in np.linspace(0, 1, n):
        px = ax + t * (bx - ax)
        py = ay + t * (by - ay)
        pz = az + t * (bz - az)
        for box in boxes:
            if pt_in_box(px, py, pz, box, mg):
                return False
    return True

def arc_clear(pts, boxes):
    arr = np.array(pts)
    for px, py, pz in arr:
        for box in boxes:
            if pt_in_box(px, py, pz, box):
                return False
    return True


def final_collision_audit(path_xyz, boxes, sample_spacing=COLLISION_SAMPLE_SPACING * 0.5, margin=0.0):
    """Dense collision check on a polyline path; returns hit count and first hit."""
    pts = np.asarray(path_xyz, dtype=float)
    if len(pts) < 2:
        return 0, None

    hits = 0
    first_hit = None

    for i in range(len(pts) - 1):
        a = pts[i]
        b = pts[i + 1]
        seg_len = float(np.linalg.norm(b - a))
        n = max(2, int(seg_len / sample_spacing) + 1)

        for t in np.linspace(0.0, 1.0, n):
            p = (1.0 - t) * a + t * b
            for bi, box in enumerate(boxes):
                if pt_in_box(float(p[0]), float(p[1]), float(p[2]), box, mg=margin):
                    hits += 1
                    if first_hit is None:
                        first_hit = (bi, i, (float(p[0]), float(p[1]), float(p[2])))
                    break

    return hits, first_hit

def add_box(ax, box, color='#922b21', edge='#e74c3c', alpha=0.55):
    x0, y0, z0, x1, y1, z1 = box
    faces = [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)],
        [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)],
    ]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors=edge, alpha=alpha, linewidths=0.6))

# ── RRT to find collision-free skeleton ────────────────────────────────────
def rrt(start_xyz, goal_xyz, boxes, n_iter=10000, step=RRT_STEP, goal_r=RRT_GOAL_RADIUS, mg=RHO+0.75):
    rng = np.random.default_rng(7)
    nodes = [np.array(start_xyz, float)]
    parent = [-1]
    goal_arr = np.array(goal_xyz, float)
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = SPACE_BOUNDS
    for _ in range(n_iter):
        s = goal_arr if rng.random() < 0.14 else np.array([
            rng.uniform(x_lo, x_hi),
            rng.uniform(y_lo, y_hi),
            rng.uniform(z_lo, z_hi),
        ])
        dists = [np.linalg.norm(n-s) for n in nodes]
        ni = int(np.argmin(dists))
        nn = nodes[ni]
        d = np.linalg.norm(s-nn)
        if d<1e-6: continue
        newp = nn + (s-nn)/d * min(step, d)
        if not seg_clear(nn[0], nn[1], nn[2], newp[0], newp[1], newp[2], boxes, mg): continue
        nodes.append(newp); parent.append(ni)
        if np.linalg.norm(newp-goal_arr)<goal_r:
            # Trace back, then append the EXACT goal point
            path = []
            idx = len(nodes)-1
            while parent[idx]!=-1:
                path.append(nodes[parent[idx]]); idx=parent[idx]
            path.reverse()
            path.append(goal_arr)   # <-- exact goal, not the nearby node
            return path
    return None

print("Running RRT...")
rrt_path = rrt(START, GOAL, BOXES)
if rrt_path is None:
    raise RuntimeError("RRT failed to find a path — try increasing n_iter")
print(f"  RRT: {len(rrt_path)} nodes")

# ── Prune collinear points ──────────────────────────────────────────────────
def prune(path, boxes, mg=RHO+0.75):
    """Remove waypoints that can be skipped without collision."""
    pruned = [path[0]]
    i = 0
    while i < len(path)-1:
        j = len(path)-1
        while j > i+1:
            if seg_clear(
                path[i][0], path[i][1], path[i][2],
                path[j][0], path[j][1], path[j][2],
                boxes, mg,
            ):
                break
            j -= 1
        pruned.append(path[j]); i = j
    return pruned

rrt_pruned = prune(rrt_path, BOXES)
print(f"  After pruning: {len(rrt_pruned)} nodes")

def particle_swarm(
    f,
    x_lb,
    x_ub,
    n=10,
    inertia=0.5,
    self_influence=1.8,
    social_influence=1.8,
    max_vel=1.0,
    f_tol=1e-6,
    max_iters=300,
    progress_every=25,
):
    global nfev
    global X_history
    nfev = 0
    X_history = []

    lhs_raw = lhsmdu.sample(len(x_lb), n)
    lhs_samples = np.array(lhs_raw).T
    X = x_lb + (x_ub - x_lb) * lhs_samples
    for i in range(n):
        X[i, :] = clip_waypoints_to_obstacle_boundary(X[i, :], x_lb, x_ub, BOXES)
    dX = (np.random.rand(n, len(x_lb))*2 - 1) * max_vel     
    X_best = X.copy()
    f_vals_best = np.full(n, np.inf)

    for i in range(n):
        fi = f(X[i, :])
        if is_collision_free_solution(X[i, :]):
            X_best[i, :] = X[i, :]
            f_vals_best[i] = fi

    n_conv = 0
    f_best_prev = np.inf
    converged = False
    last_iter = 0
    for k in range(1, max_iters + 1):
        last_iter = k
        f_vals = np.array([f(xi) for xi in X])
        feasible = np.array([is_collision_free_solution(xi) for xi in X], dtype=bool)
        for i in range(n):
            if feasible[i] and f_vals[i] < f_vals_best[i]:
                X_best[i,:] = X[i,:]
                f_vals_best[i] = f_vals[i]

        if np.any(feasible):
            feasible_scores = np.where(feasible, f_vals, np.inf)
            swarm_best = X[int(np.argmin(feasible_scores)), :].copy()
        elif np.any(np.isfinite(f_vals_best)):
            swarm_best = X_best[int(np.argmin(f_vals_best)), :].copy()
        else:
            swarm_best = np.mean(X, axis=0)

        for i in range(n):
            dX[i,:] = inertia*dX[i,:] + self_influence*(X_best[i,:]-X[i,:]) + social_influence*(swarm_best - X[i,:])
            dX[i,:] = np.maximum(np.minimum(dX[i,:], max_vel),-max_vel)
            X[i,:] += dX[i,:]
            X[i,:] = np.maximum(np.minimum(X[i,:], x_ub), x_lb)
            projected = clip_waypoints_to_obstacle_boundary(X[i, :], x_lb, x_ub, BOXES)
            if not np.allclose(projected, X[i, :]):
                dX[i, :] *= 0.5
                X[i, :] = projected
        
            for j in range(len(x_lb)):
                if X[i,j] < x_lb[j]:
                    X[i,j] = x_lb[j]
                    dX[i,j] *= -0.5

                elif X[i,j] > x_ub[j]:
                    X[i,j] = x_ub[j]
                    dX[i,j] *= -0.5

        finite_best = f_vals_best[np.isfinite(f_vals_best)]
        if finite_best.size == 0:
            if progress_every > 0 and (k % progress_every == 0):
                print(f"PSO iter {k}: no feasible particles yet")
            n_conv = 0
            X_history.append(X.copy())
            max_vel *= 0.9
            continue

        f_min = np.min(finite_best)

        if np.abs(f_best_prev - f_min) < f_tol:
            n_conv += 1
            if n_conv > 2:
                converged = True
        else:
            n_conv = 0

        f_best_prev = f_min

        X_history.append(X.copy())
        max_vel *= 0.9

        if progress_every > 0 and (k % progress_every == 0):
            print(f"PSO iter {k}: best feasible cost = {f_min:.2f}")

        if converged:
            break

    if not np.any(np.isfinite(f_vals_best)):
        raise RuntimeError(
            f"PSO did not find a collision-free solution after {last_iter} iterations. "
            "Try increasing n, max_iters, or LOCAL_BOUNDS_MARGIN."
        )

    if not converged:
        print(f"PSO reached max_iters={max_iters}; returning best feasible particle found.")

    r_x = X_best[int(np.argmin(f_vals_best))]
    return r_x, f(r_x), nfev


def particle_swarm_legacy(f, x_lb, x_ub, n=20, inertia=0.5, self_influence=1.8, social_influence=1.8, max_vel=5.0, f_tol=1e-6, max_iters=300):
    """Earlier baseline PSO update logic for comparison."""
    global nfev
    global X_history
    nfev = 0
    X_history = []

    lhs_raw = lhsmdu.sample(len(x_lb), n)
    lhs_samples = np.array(lhs_raw).T
    X = x_lb + (x_ub - x_lb) * lhs_samples
    dX = (np.random.rand(n, len(x_lb)) * 2 - 1) * max_vel
    X_best = X.copy()
    f_vals_best = np.array([f(xi) for xi in X_best])

    n_conv = 0
    f_best_prev = np.inf
    converged = False

    for _ in range(max_iters):
        f_vals = np.array([f(xi) for xi in X])
        swarm_best_i = int(np.argmin(f_vals))

        for i in range(n):
            if f_vals[i] < f_vals_best[i]:
                X_best[i, :] = X[i, :]
                f_vals_best[i] = f_vals[i]

        swarm_best = X[swarm_best_i, :].copy()

        for i in range(n):
            dX[i, :] = inertia * dX[i, :] + self_influence * (X_best[i, :] - X[i, :]) + social_influence * (swarm_best - X[i, :])
            dX[i, :] = np.maximum(np.minimum(dX[i, :], max_vel), -max_vel)
            X[i, :] += dX[i, :]
            X[i, :] = np.maximum(np.minimum(X[i, :], x_ub), x_lb)

        f_min = float(np.min(f_vals_best))
        if abs(f_best_prev - f_min) < f_tol:
            n_conv += 1
            if n_conv > 2:
                converged = True
                break
        else:
            n_conv = 0
        f_best_prev = f_min
        X_history.append(X.copy())
        max_vel *= 0.9

    if not converged:
        print("Legacy PSO reached max_iters; returning best particle found.")

    r_x = X_best[int(np.argmin(f_vals_best))]
    return r_x, f(r_x), nfev


def evaluate_candidate(x):
    x_s, y_s, z_s, length, curvature, _ = build_spline_from_x(x)
    path_curve = np.column_stack((x_s, y_s, z_s))
    exact_hits, _ = final_collision_audit(path_curve, BOXES, margin=0.0)
    margin_hits, _ = final_collision_audit(path_curve, BOXES, margin=IPM_ACCEPT_MARGIN)
    max_curvature = float(np.max(curvature))
    return {
        "objective": float(objective(x)),
        "length": float(length),
        "max_curvature": max_curvature,
        "max_curvature_violation": float(max(0.0, max_curvature - (1.0 / RHO))),
        "exact_hits": int(exact_hits),
        "margin_hits": int(margin_hits),
        "x_s": x_s,
        "y_s": y_s,
        "z_s": z_s,
        "curvature": curvature,
        "path_curve": path_curve,
    }


def visualize_swarm(X_history, x_lb, x_ub, title="PSO Animation"):

    fig, ax = plt.subplots()
    ax.set_xlim(x_lb[0], x_ub[0])
    ax.set_ylim(x_lb[1], x_ub[1])

    n_particles = X_history[0].shape[0]

    # scatter for particles
    scatter = ax.scatter([], [], s=50)

    # one line per particle
    lines = [ax.plot([], [], lw=1)[0] for _ in range(n_particles)]

    # egg carton contour (fast vectorized form)
    x = np.linspace(x_lb[0], x_ub[0], 100)
    y = np.linspace(x_lb[1], x_ub[1], 100)
    Xg, Yg = np.meshgrid(x, y)
    Z = 0.1*Xg**2 + 0.1*Yg**2 - np.cos(3*Xg) - np.cos(3*Yg)

    ax.contour(Xg, Yg, Z, levels=20, alpha=0.5)

    def update(frame):

        X = X_history[frame]
        scatter.set_offsets(X)

        if frame > 0:
            X_prev = X_history[frame-1]

            for i, line in enumerate(lines):
                line.set_data(
                    [X_prev[i,0], X[i,0]],
                    [X_prev[i,1], X[i,1]]
                )

        ax.set_title(f"Particle Swarm")

        return [scatter] + lines

    anim = FuncAnimation(
        fig,
        update,
        frames=len(X_history),
        interval=200,
        blit=True
    )

    plt.show()


def visualize_path_evolution(X_history, objective_fn, output_path, max_frames=80, fps=6):
    """Animate best PSO path across iterations and save as GIF."""
    if not X_history:
        return

    stride = max(1, len(X_history) // max_frames)
    sampled_frames = list(range(0, len(X_history), stride))

    best_x_per_frame = []
    best_cost_per_frame = []
    for frame_idx in sampled_frames:
        swarm = X_history[frame_idx]
        costs = np.array([objective_fn(xi) for xi in swarm])
        j = int(np.argmin(costs))
        best_x_per_frame.append(swarm[j].copy())
        best_cost_per_frame.append(float(costs[j]))

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0d0d1a')

    for box in BOXES:
        add_box(ax, box, alpha=0.28)

    ax.plot([START[0]], [START[1]], [START[2]], 'o', color='#2ecc71', ms=10)
    ax.plot([GOAL[0]], [GOAL[1]], [GOAL[2]], '*', color='#e74c3c', ms=14)

    ax.set_xlim(SPACE_BOUNDS[0], SPACE_BOUNDS[1])
    ax.set_ylim(SPACE_BOUNDS[2], SPACE_BOUNDS[3])
    ax.set_zlim(SPACE_BOUNDS[4], SPACE_BOUNDS[5])
    ax.set_box_aspect((1, 1, 0.55))
    ax.view_init(elev=24, azim=-148)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Draw a bright underlay plus a cyan overlay so the path remains visible through boxes.
    line_glow, = ax.plot([], [], [], color='white', lw=4.6, alpha=0.90)
    line, = ax.plot([], [], [], color='#00d1ff', lw=2.6)
    ctrl_poly, = ax.plot([], [], [], color='#ffd166', lw=1.8, alpha=0.95)
    ctrl = ax.scatter([], [], [], c='#ffdd57', s=42, edgecolors='black', linewidths=0.6, depthshade=False)

    def update(frame_i):
        x = best_x_per_frame[frame_i]
        x_s, y_s, z_s, _, _, _ = build_spline_from_x(x)
        line_glow.set_data(x_s, y_s)
        line_glow.set_3d_properties(z_s)
        line.set_data(x_s, y_s)
        line.set_3d_properties(z_s)

        wps = np.array([(x[k], x[k + 1], x[k + 2]) for k in range(0, len(x), 3)], dtype=float)
        if len(wps) > 0:
            ctrl._offsets3d = (wps[:, 0], wps[:, 1], wps[:, 2])
            ctrl_pts = np.vstack((np.array(START), wps, np.array(GOAL)))
            ctrl_poly.set_data(ctrl_pts[:, 0], ctrl_pts[:, 1])
            ctrl_poly.set_3d_properties(ctrl_pts[:, 2])
        else:
            ctrl._offsets3d = ([], [], [])
            ctrl_poly.set_data([], [])
            ctrl_poly.set_3d_properties([])

        ax.set_title(f"PSO Path Evolution | frame {frame_i + 1}/{len(best_x_per_frame)} | cost={best_cost_per_frame[frame_i]:.1f}")
        return line_glow, line, ctrl_poly, ctrl

    anim = FuncAnimation(fig, update, frames=len(best_x_per_frame), interval=int(1000 / fps), blit=False)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def visualize_agent_traverse(path_xyz, output_path, max_frames=120, fps=8):
    """Animate a moving agent along the final path with a chase camera."""
    if len(path_xyz) < 2:
        return

    path_xyz = np.asarray(path_xyz, dtype=float)
    segments = path_xyz[1:] - path_xyz[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    total_length = float(np.sum(lengths))
    if total_length <= 1e-9:
        return

    samples_per_seg = np.maximum(2, np.ceil(max_frames * (lengths / total_length)).astype(int))
    pts = []
    seg_ids = []
    for i, n_samples in enumerate(samples_per_seg):
        for t in np.linspace(0.0, 1.0, int(n_samples), endpoint=False):
            pts.append((1.0 - t) * path_xyz[i] + t * path_xyz[i + 1])
            seg_ids.append(i)
    pts.append(path_xyz[-1])
    seg_ids.append(len(path_xyz) - 2)
    pts = np.asarray(pts)

    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0d0d1a')

    for box in BOXES:
        add_box(ax, box, alpha=0.22)

    ax.plot(path_xyz[:, 0], path_xyz[:, 1], path_xyz[:, 2], color='#5bc0eb', lw=2.1, alpha=0.45)
    ax.plot([START[0]], [START[1]], [START[2]], 'o', color='#2ecc71', ms=10)
    ax.plot([GOAL[0]], [GOAL[1]], [GOAL[2]], '*', color='#e74c3c', ms=14)

    agent = ax.scatter([], [], [], s=130, c='#ffdd57', edgecolors='black', linewidths=1.0, depthshade=False)
    trail, = ax.plot([], [], [], color='#00d1ff', lw=3.0)
    last_heading = [0.0]
    current_azim = [180.0]

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect((1, 1, 0.55))

    def update(frame_i):
        pos = pts[frame_i]
        history = pts[: frame_i + 1]
        agent._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
        trail.set_data(history[:, 0], history[:, 1])
        trail.set_3d_properties(history[:, 2])

        lookback = pts[max(frame_i - 4, 0)]
        direction = pos - lookback
        if np.linalg.norm(direction[:2]) > 1e-6:
            last_heading[0] = float(np.degrees(np.arctan2(direction[1], direction[0])))

        center = pos
        span = max(150.0, 0.15 * WORLD_SCALE)
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(max(SPACE_BOUNDS[4], center[2] - 0.45 * span), min(SPACE_BOUNDS[5], center[2] + 0.45 * span))
        target_azim = last_heading[0] + 180.0
        delta = ((target_azim - current_azim[0] + 180.0) % 360.0) - 180.0
        current_azim[0] += 0.16 * delta
        ax.view_init(elev=36, azim=current_azim[0])
        ax.set_title(f"Agent Traverse | frame {frame_i + 1}/{len(pts)}")
        return agent, trail

    anim = FuncAnimation(fig, update, frames=len(pts), interval=int(1000 / fps), blit=False)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

def build_spline_from_x(x, n_samples=200):
    pts = [START[:3]]

    for i in range(0, len(x), 3):
        pts.append((x[i], x[i+1], x[i+2]))

    pts.append(GOAL[:3])

    x_s, y_s, z_s, dx, dy, dz, ddx, ddy, ddz = build_bspline(pts, n_samples=n_samples)

    curvature = compute_curvature(dx, dy, dz, ddx, ddy, ddz)

    # arc length
    ds = np.sqrt(dx**2 + dy**2 + dz**2)
    length = np.sum(ds)

    # collision
    collision = False
    for px, py, pz in zip(x_s, y_s, z_s):
        for box in BOXES:
            if pt_in_box(px, py, pz, box, mg=COLLISION_CLEARANCE):
                collision = True

    return x_s, y_s, z_s, length, curvature, collision


def clip_waypoints_to_obstacle_boundary(x, x_lb, x_ub, boxes, pushout=1.0):
    arr = np.array(x, dtype=float, copy=True)
    arr = np.maximum(np.minimum(arr, x_ub), x_lb)

    if len(arr) % 3 != 0:
        return arr

    for i in range(0, len(arr), 3):
        p = arr[i:i + 3].copy()
        for _ in range(6):
            moved = False
            for box in boxes:
                if pt_in_box(p[0], p[1], p[2], box, mg=0.0):
                    distances = np.array([
                        p[0] - box[0],
                        box[3] - p[0],
                        p[1] - box[1],
                        box[4] - p[1],
                        p[2] - box[2],
                        box[5] - p[2],
                    ], dtype=float)
                    side = int(np.argmin(distances))
                    if side == 0:
                        p[0] = box[0] - pushout
                    elif side == 1:
                        p[0] = box[3] + pushout
                    elif side == 2:
                        p[1] = box[1] - pushout
                    elif side == 3:
                        p[1] = box[4] + pushout
                    elif side == 4:
                        p[2] = box[2] - pushout
                    else:
                        p[2] = box[5] + pushout
                    p = np.maximum(np.minimum(p, x_ub[i:i + 3]), x_lb[i:i + 3])
                    moved = True
            if not moved:
                break
        arr[i:i + 3] = p

    return arr


def is_collision_free_solution(x):
    x_s, y_s, z_s, _, _, collision = build_spline_from_x(x, n_samples=COLLISION_CONSTRAINT_SAMPLES)
    if collision:
        return False
    path_curve = np.column_stack((x_s, y_s, z_s))
    hits, _ = final_collision_audit(path_curve, BOXES, margin=COLLISION_CLEARANCE)
    return hits == 0

def objective(x):
    x_s, y_s, z_s, length, curvature, collision = build_spline_from_x(x)

    # curvature constraint
    kappa_max = 1.0 / RHO
    curvature_violation = np.maximum(0, curvature - kappa_max)

    curvature_penalty = np.sum(curvature_violation**2) * 1e4

    path_curve = np.column_stack((x_s, y_s, z_s))
    hits, _ = final_collision_audit(path_curve, BOXES, margin=0.0)
    collision_penalty = 1e6 if (hits > 0 or collision) else 0

    return length + curvature_penalty + collision_penalty


def point_box_signed_distance(point_xyz, box):
    """Signed distance to an axis-aligned box: positive outside, negative inside."""
    x, y, z = float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])
    x0, y0, z0, x1, y1, z1 = box

    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    dz = max(z0 - z, 0.0, z - z1)
    outside = math.sqrt(dx * dx + dy * dy + dz * dz)
    if outside > 0.0:
        return outside

    inside_margin = min(x - x0, x1 - x, y - y0, y1 - y, z - z0, z1 - z)
    return -inside_margin


def collision_constraint_values(x, boxes, clearance=COLLISION_CLEARANCE, n_checks=COLLISION_CONSTRAINT_SAMPLES):
    """Inequality values that must stay >= 0 for collision-free spline samples."""
    x_s, y_s, z_s, _, _, _ = build_spline_from_x(x, n_samples=n_checks)
    total = len(x_s)
    if total == 0:
        return np.array([], dtype=float)

    n_checks = max(2, min(int(n_checks), total))
    indices = np.linspace(0, total - 1, n_checks, dtype=int)

    values = []
    for idx in indices:
        p = (x_s[idx], y_s[idx], z_s[idx])
        min_dist = min(point_box_signed_distance(p, box) for box in boxes)
        values.append(min_dist - clearance)
    return np.asarray(values, dtype=float)


def densify_path_nodes(path_nodes, refinement_level):
    nodes = [np.array(p, dtype=float) for p in path_nodes]
    for _ in range(refinement_level):
        dense = [nodes[0]]
        for a, b in zip(nodes[:-1], nodes[1:]):
            dense.append(0.5 * (a + b))
            dense.append(b)
        nodes = dense
    return [tuple(p) for p in nodes]


def build_bounds_and_initial_guess(opt_nodes, margin):
    lb_local = []
    ub_local = []
    x0_local = []

    for (x, y, z) in opt_nodes:
        lb_local.extend([
            max(SPACE_BOUNDS[0], x - margin),
            max(SPACE_BOUNDS[2], y - margin),
            max(SPACE_BOUNDS[4], z - margin),
        ])
        ub_local.extend([
            min(SPACE_BOUNDS[1], x + margin),
            min(SPACE_BOUNDS[3], y + margin),
            min(SPACE_BOUNDS[5], z + margin),
        ])
        x0_local.extend([x, y, z])

    lb_local = np.array(lb_local, dtype=float)
    ub_local = np.array(ub_local, dtype=float)
    x0_local = np.array(x0_local, dtype=float)
    x0_local = clip_waypoints_to_obstacle_boundary(x0_local, lb_local, ub_local, BOXES)
    return lb_local, ub_local, x0_local

# ── Build bounds from RRT path ─────────────────────────────────────────────
margin = LOCAL_BOUNDS_MARGIN


def summarize_solution(label, x_sol):
    x_s, y_s, z_s, length, curvature, _ = build_spline_from_x(x_sol)
    path_curve = np.column_stack((x_s, y_s, z_s))
    exact_hits, exact_first = final_collision_audit(path_curve, BOXES, margin=0.0)
    buffer_hits, buffer_first = final_collision_audit(path_curve, BOXES, margin=COLLISION_CLEARANCE)

    print(f"{label} length: {length:.2f}")
    print(f"{label} max curvature: {float(np.max(curvature)):.6f}")
    print(f"{label} collision (exact): {'COLLISION' if exact_hits > 0 else 'clear'} | hits={exact_hits}")
    if exact_first is not None:
        bi, seg_i, p = exact_first
        print(f"  {label} first exact hit: box={bi}, segment={seg_i}, point=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")
    print(f"{label} collision ({COLLISION_CLEARANCE:.1f} m buffer): {'COLLISION' if buffer_hits > 0 else 'clear'} | hits={buffer_hits}")
    if buffer_first is not None:
        bi, seg_i, p = buffer_first
        print(f"  {label} first margin hit: box={bi}, segment={seg_i}, point=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")

    return {
        "x": x_sol,
        "x_s": x_s,
        "y_s": y_s,
        "z_s": z_s,
        "length": float(length),
        "curvature": curvature,
        "path_curve": path_curve,
    }


def clone_swarm_history(history):
    return [np.array(frame, dtype=float).copy() for frame in history]


def score_candidate_eval(eval_data):
    return (
        int(eval_data["exact_hits"]),
        int(eval_data.get("margin_hits", 0)),
        float(eval_data["objective"]),
    )


def history_metrics_from_swarm(history):
    length_history = []
    curvature_history = []
    hit_history = []
    for swarm in history:
        candidates = [evaluate_candidate(np.array(p, dtype=float)) for p in swarm]
        best_eval = min(candidates, key=score_candidate_eval)
        length_history.append(float(best_eval["length"]))
        curvature_history.append(float(best_eval["max_curvature"]))
        hit_history.append(int(best_eval["exact_hits"]))
    return length_history, curvature_history, hit_history


def history_metrics_from_states(history):
    length_history = []
    curvature_history = []
    hit_history = []
    for x in history:
        eval_data = evaluate_candidate(np.array(x, dtype=float))
        length_history.append(float(eval_data["length"]))
        curvature_history.append(float(eval_data["max_curvature"]))
        hit_history.append(int(eval_data["exact_hits"]))
    return length_history, curvature_history, hit_history



# ── GCS Planner Integration ──────────────────────────────────────────────
def point_inside_obstacle(x, y, z, boxes):
    return any(box[0] <= x <= box[3] and box[1] <= y <= box[4] and box[2] <= z <= box[5] for box in boxes)

def seg_clear_gcs(a, b, boxes, mg=0.3):
    n = max(20, int(np.linalg.norm(np.asarray(b) - np.asarray(a)) / COLLISION_SAMPLE_SPACING))
    for t in np.linspace(0.0, 1.0, n):
        p = (1.0 - t) * np.asarray(a) + t * np.asarray(b)
        px, py, pz = float(p[0]), float(p[1]), float(p[2])
        for box in boxes:
            if (
                box[0] - mg <= px <= box[3] + mg
                and box[1] - mg <= py <= box[4] + mg
                and box[2] - mg <= pz <= box[5] + mg
            ):
                return False
    return True

def point_in_box_margin(point_xyz, box, margin=0.0):
    x, y, z = float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])
    return (
        box[0] - margin <= x <= box[3] + margin
        and box[1] - margin <= y <= box[4] + margin
        and box[2] - margin <= z <= box[5] + margin
    )

def final_collision_audit_gcs(path_xyz, boxes, sample_spacing=COLLISION_SAMPLE_SPACING * 0.5, margin=0.0):
    pts = np.asarray(path_xyz, dtype=float)
    if len(pts) < 2:
        return 0, None
    hits = 0
    first_hit = None
    for i in range(len(pts) - 1):
        a = pts[i]
        b = pts[i + 1]
        seg_len = float(np.linalg.norm(b - a))
        n = max(2, int(seg_len / sample_spacing) + 1)
        for t in np.linspace(0.0, 1.0, n):
            p = (1.0 - t) * a + t * b
            for bi, box in enumerate(boxes):
                if point_in_box_margin(p, box, margin=margin):
                    hits += 1
                    if first_hit is None:
                        first_hit = (bi, i, (float(p[0]), float(p[1]), float(p[2])))
                    break
    return hits, first_hit

def build_free_boxes(boxes, bounds):
    x0, x1, y0, y1, z0, z1 = bounds
    x_cuts = sorted({x0, x1, *[b[0] for b in boxes], *[b[3] for b in boxes]})
    y_cuts = sorted({y0, y1, *[b[1] for b in boxes], *[b[4] for b in boxes]})
    z_cuts = sorted({z0, z1, *[b[2] for b in boxes], *[b[5] for b in boxes]})
    cells = {}
    node_ids = {}
    nodes = []
    idx_to_ijk = {}
    for i in range(len(x_cuts) - 1):
        for j in range(len(y_cuts) - 1):
            for k in range(len(z_cuts) - 1):
                cx0, cx1 = x_cuts[i], x_cuts[i + 1]
                cy0, cy1 = y_cuts[j], y_cuts[j + 1]
                cz0, cz1 = z_cuts[k], z_cuts[k + 1]
                if cx1 - cx0 <= 1e-9 or cy1 - cy0 <= 1e-9 or cz1 - cz0 <= 1e-9:
                    continue
                mx, my, mz = 0.5 * (cx0 + cx1), 0.5 * (cy0 + cy1), 0.5 * (cz0 + cz1)
                if not point_inside_obstacle(mx, my, mz, boxes):
                    idx = len(nodes)
                    cells[(i, j, k)] = (cx0, cx1, cy0, cy1, cz0, cz1)
                    node_ids[(i, j, k)] = idx
                    idx_to_ijk[idx] = (i, j, k)
                    nodes.append((cx0, cx1, cy0, cy1, cz0, cz1))
    return nodes, node_ids, idx_to_ijk

def cell_center(cell):
    return np.array([(cell[0] + cell[1]) * 0.5, (cell[2] + cell[3]) * 0.5, (cell[4] + cell[5]) * 0.5])

def find_cell_for_point(point_xyz, cells):
    x, y, z = point_xyz
    for idx, (x0, x1, y0, y1, z0, z1) in enumerate(cells):
        if x0 - 1e-9 <= x <= x1 + 1e-9 and y0 - 1e-9 <= y <= y1 + 1e-9 and z0 - 1e-9 <= z <= z1 + 1e-9:
            return idx
    return None

def build_adjacency(cells, node_ids, idx_to_ijk):
    neighbors = {i: [] for i in range(len(cells))}
    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for idx, (i, j, k) in idx_to_ijk.items():
        for di, dj, dk in dirs:
            nxt = (i + di, j + dj, k + dk)
            if nxt in node_ids:
                neighbors[idx].append(node_ids[nxt])
    return neighbors

def dijkstra(cells, neighbors, start_idx, goal_idx):
    dist = {start_idx: 0.0}
    prev = {}
    pq = [(0.0, start_idx)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == goal_idx:
            break
        if d > dist.get(u, float("inf")):
            continue
        cu = cell_center(cells[u])
        for v in neighbors[u]:
            cv = cell_center(cells[v])
            w = float(np.linalg.norm(cv - cu))
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if goal_idx not in dist:
        return None
    path = [goal_idx]
    while path[-1] != start_idx:
        path.append(prev[path[-1]])
    path.reverse()
    return path

def portal_bounds(a, b):
    ax0, ax1, ay0, ay1, az0, az1 = a
    bx0, bx1, by0, by1, bz0, bz1 = b
    same_x = abs(ax1 - bx0) <= 1e-9 or abs(bx1 - ax0) <= 1e-9
    same_y = abs(ay1 - by0) <= 1e-9 or abs(by1 - ay0) <= 1e-9
    same_z = abs(az1 - bz0) <= 1e-9 or abs(bz1 - az0) <= 1e-9
    if same_x:
        x = ax1 if abs(ax1 - bx0) <= 1e-9 else bx1
        y0, y1 = max(ay0, by0), min(ay1, by1)
        z0, z1 = max(az0, bz0), min(az1, bz1)
        return (x, x, y0, y1, z0, z1)
    if same_y:
        y = ay1 if abs(ay1 - by0) <= 1e-9 else by1
        x0, x1 = max(ax0, bx0), min(ax1, bx1)
        z0, z1 = max(az0, bz0), min(az1, bz1)
        return (x0, x1, y, y, z0, z1)
    if same_z:
        z = az1 if abs(az1 - bz0) <= 1e-9 else bz1
        x0, x1 = max(ax0, bx0), min(ax1, bx1)
        y0, y1 = max(ay0, by0), min(ay1, by1)
        return (x0, x1, y0, y1, z, z)
    raise RuntimeError("Cells are not adjacent with a shared portal face.")

def polyline_length(path_xyz):
    return float(sum(np.linalg.norm(np.asarray(path_xyz[i + 1]) - np.asarray(path_xyz[i])) for i in range(len(path_xyz) - 1)))

def optimize_portals(start_xyz, goal_xyz, portals, return_history=False):
    if not portals:
        return ([], {"objective": [], "length": [], "violation": []}) if return_history else []
    mids = [np.array([(p[0] + p[1]) * 0.5, (p[2] + p[3]) * 0.5, (p[4] + p[5]) * 0.5], dtype=float) for p in portals]
    max_optimized_portals = 16
    if len(portals) <= max_optimized_portals:
        opt_indices = list(range(len(portals)))
    else:
        opt_indices = sorted(set(np.linspace(0, len(portals) - 1, max_optimized_portals, dtype=int).tolist()))
    init = []
    bounds = []
    for i in opt_indices:
        px0, px1, py0, py1, pz0, pz1 = portals[i]
        init.extend([(px0 + px1) * 0.5, (py0 + py1) * 0.5, (pz0 + pz1) * 0.5])
        bounds.extend([(px0, px1), (py0, py1), (pz0, pz1)])
    def unpack(flat):
        wps = [m.copy() for m in mids]
        for k, i in enumerate(opt_indices):
            wps[i] = np.array([flat[3 * k], flat[3 * k + 1], flat[3 * k + 2]], dtype=float)
        return wps
    history = {"objective": [], "length": [], "curvature": [], "hits": []}
    def evaluate_flat(flat):
        wps = unpack(flat)
        pts = [np.array(start_xyz, dtype=float), *wps, np.array(goal_xyz, dtype=float)]
        length = float(sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)))
        x, y, z, dx, dy, dz, ddx, ddy, ddz = build_bspline(pts)
        curvature = compute_curvature(dx, dy, dz, ddx, ddy, ddz)
        kappa_max = 1.0 / RHO
        curvature_violation = np.maximum(0.0, curvature - kappa_max)
        curvature_penalty = float(np.sum(curvature_violation ** 2) * 1e4)
        path_curve = np.column_stack((x, y, z))
        hits, _ = final_collision_audit_gcs(path_curve, BOXES, margin=0.0)
        return {
            "pts": pts,
            "length": length,
            "curvature": curvature,
            "curvature_penalty": curvature_penalty,
            "objective": length + curvature_penalty,
            "hits": int(hits),
        }
    def objective(flat):
        return evaluate_flat(flat)["objective"]
    def callback(flat):
        metrics = evaluate_flat(flat)
        history["objective"].append(metrics["objective"])
        history["length"].append(metrics["length"])
        history["curvature"].append(float(np.max(metrics["curvature"])))
        history["hits"].append(int(metrics["hits"]))
    init = np.array(init, dtype=float)
    initial_metrics = evaluate_flat(init)
    history["objective"].append(initial_metrics["objective"])
    history["length"].append(initial_metrics["length"])
    history["curvature"].append(float(np.max(initial_metrics["curvature"])))
    history["hits"].append(int(initial_metrics["hits"]))
    res = minimize(
        objective,
        init,
        method="L-BFGS-B",
        bounds=bounds,
        callback=callback,
        options={"maxiter": 120, "maxfun": 4000, "ftol": 1e-6},
    )
    if res.success:
        center = np.array(res.x, dtype=float)
        local_bounds = []
        for idx, (lo, hi) in enumerate(bounds):
            span = hi - lo
            radius = 0.18 * span
            local_bounds.append((max(lo, center[idx] - radius), min(hi, center[idx] + radius)))
        res2 = minimize(
            objective,
            center,
            method="L-BFGS-B",
            bounds=local_bounds,
            callback=callback,
            options={"maxiter": 60, "maxfun": 2000, "ftol": 1e-7},
        )
        if res2.success and res2.fun <= res.fun:
            res = res2
    if not res.success:
        return (mids, history) if return_history else mids
    result = unpack(res.x)
    final_metrics = evaluate_flat(np.array(res.x, dtype=float))
    if not history["length"] or abs(history["length"][-1] - final_metrics["length"]) > 1e-9:
        history["objective"].append(final_metrics["objective"])
        history["length"].append(final_metrics["length"])
        history["curvature"].append(float(np.max(final_metrics["curvature"])))
        history["hits"].append(int(final_metrics["hits"]))
    return (result, history) if return_history else result

def run_gcs_planner(plot=True, save_plot=True, save_gif=False):
    rng = np.random.default_rng(7)
    start = sample_free_point(
        rng,
        BOXES,
        (SPACE_BOUNDS[0], SPACE_BOUNDS[1]),
        (SPACE_BOUNDS[2], SPACE_BOUNDS[3]),
        (START_Z, START_Z),
        xy_margin=0.05 * WORLD_SCALE,
        z_margin=0.0,
    )
    GOAL_Z = 0.5 * SPACE_BOUNDS[5]
    goal = sample_free_point(
        rng,
        BOXES,
        (SPACE_BOUNDS[0], SPACE_BOUNDS[1]),
        (SPACE_BOUNDS[2], SPACE_BOUNDS[3]),
        (GOAL_Z, GOAL_Z),
        xy_margin=0.05 * WORLD_SCALE,
        z_margin=0.0,
    )
    while math.dist(start, goal) < 0.20 * WORLD_SCALE:
        goal = sample_free_point(
            rng,
            BOXES,
            (SPACE_BOUNDS[0], SPACE_BOUNDS[1]),
            (SPACE_BOUNDS[2], SPACE_BOUNDS[3]),
            (GOAL_Z, GOAL_Z),
            xy_margin=0.05 * WORLD_SCALE,
            z_margin=0.0,
        )
    print(f"Start: {start}")
    print(f"Goal:  {goal}")
    print("Building 3D convex decomposition...")
    cells, node_ids, idx_to_ijk = build_free_boxes(BOXES, SPACE_BOUNDS)
    neighbors = build_adjacency(cells, node_ids, idx_to_ijk)
    s_idx = find_cell_for_point(start, cells)
    g_idx = find_cell_for_point(goal, cells)
    if s_idx is None or g_idx is None:
        raise RuntimeError("Start or goal is not in any free convex cell.")
    print(f"  Convex cells: {len(cells)}")
    print("Solving graph path over convex sets...")
    cell_path = dijkstra(cells, neighbors, s_idx, g_idx)
    if cell_path is None:
        raise RuntimeError("No path exists through convex-set graph.")
    print(f"  Sets used in path: {len(cell_path)}")
    portals = []
    for a, b in zip(cell_path[:-1], cell_path[1:]):
        portals.append(portal_bounds(cells[a], cells[b]))
    print("Optimizing portal points (continuous convex step)...")
    wps, portal_history = optimize_portals(np.array(start, dtype=float), np.array(goal, dtype=float), portals, return_history=True)
    path_xyz = [np.array(start, dtype=float), *wps, np.array(goal, dtype=float)]
    clear = all(seg_clear_gcs(path_xyz[i], path_xyz[i + 1], BOXES) for i in range(len(path_xyz) - 1))
    length = polyline_length(path_xyz)
    x_s, y_s, z_s, dx, dy, dz, ddx, ddy, ddz = build_bspline(path_xyz)
    curvature = compute_curvature(dx, dy, dz, ddx, ddy, ddz)
    kappa_max = 1.0 / RHO
    z_min = float(np.min(z_s))
    z_max = float(np.max(z_s))
    path_curve = np.column_stack((x_s, y_s, z_s))
    print(f"Length: {length:.2f}")
    print(f"Collision free: {clear}")
    print(f"Max curvature: {float(np.max(curvature)):.6f}")
    print(f"Allowed: {kappa_max:.6f}")
    print(f"Spline z-range: [{z_min:.2f}, {z_max:.2f}]")
    exact_hits, exact_first = final_collision_audit_gcs(path_curve, BOXES, margin=0.0)
    buffer_hits, buffer_first = final_collision_audit_gcs(path_curve, BOXES, margin=0.3)
    print(f"Final collision check (exact boxes): {'COLLISION' if exact_hits > 0 else 'clear'} | hits={exact_hits}")
    if exact_first is not None:
        bi, seg_i, p = exact_first
        print(f"  First exact hit: box={bi}, segment={seg_i}, point=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")
    print(f"Final collision check (with 0.3 margin): {'COLLISION' if buffer_hits > 0 else 'clear'} | hits={buffer_hits}")
    if buffer_first is not None:
        bi, seg_i, p = buffer_first
        print(f"  First margin hit: box={bi}, segment={seg_i}, point=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")
    plot_path = os.path.join(os.path.dirname(__file__), "pathfinder_gcs_3d.png")
    if plot:
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#0d0d1a")
        for box in BOXES:
            add_box(ax, box)
        arr = np.asarray(path_xyz)
        ax.plot(x_s, y_s, z_s, color="#00d1ff", lw=2.8)
        ax.scatter(x_s, y_s, z_s, c=curvature, cmap="plasma", s=8, alpha=0.85)
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], c="yellow", s=28, edgecolors="white", depthshade=False)
        ax.plot([start[0]], [start[1]], [start[2]], "o", color="#2ecc71", ms=10)
        ax.plot([goal[0]], [goal[1]], [goal[2]], "*", color="#e74c3c", ms=14)
        ax.set_xlim(SPACE_BOUNDS[0], SPACE_BOUNDS[1])
        ax.set_ylim(SPACE_BOUNDS[2], SPACE_BOUNDS[3])
        ax.set_zlim(SPACE_BOUNDS[4], SPACE_BOUNDS[5])
        ax.set_box_aspect((1, 1, 0.55))
        ax.view_init(elev=24, azim=-58)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title(f"3D GCS Path | Length={length:.1f} | max k={float(np.max(curvature)):.3f}")
        if save_plot:
            plt.savefig(plot_path, dpi=200, bbox_inches="tight")
            print(f"Saved plot to {plot_path}")
        plt.show()
    return {
        "start": start,
        "goal": goal,
        "path_xyz": path_xyz,
        "path_curve": path_curve,
        "x_s": x_s,
        "y_s": y_s,
        "z_s": z_s,
        "length": length,
        "curvature": curvature,
        "max_curvature": float(np.max(curvature)),
        "exact_hits": int(exact_hits),
        "buffer_hits": int(buffer_hits),
        "length_history": list(portal_history["length"]),
        "curvature_history": list(portal_history["curvature"]),
        "hit_history": list(portal_history["hits"]),
        "objective_history": list(portal_history["objective"]),
        "plot_path": plot_path,
    }

# ── Run PSO with RRT-seeded feasible initialization ────────────────────────
rrt_seed_candidates = [rrt_pruned[1:-1], rrt_path[1:-1]]
seed_name = None
opt_nodes = None
lb = None
ub = None
x0 = None

for name, candidate_nodes in [("pruned RRT", rrt_seed_candidates[0]), ("full RRT", rrt_seed_candidates[1])]:
    if len(candidate_nodes) == 0:
        continue
    lb_c, ub_c, x0_c = build_bounds_and_initial_guess(candidate_nodes, margin)
    if is_collision_free_solution(x0_c):
        seed_name = name
        opt_nodes = candidate_nodes
        lb, ub, x0 = lb_c, ub_c, x0_c
        break

if x0 is None:
    raise RuntimeError("No feasible RRT-seeded initial solution found. Try increasing RRT n_iter or LOCAL_BOUNDS_MARGIN.")

print(f"Optimization dim: {len(lb)}")
print(f"Using feasible initial seed from {seed_name} with {len(opt_nodes)} waypoints")


results = {}

if RUN_PSO_CURRENT:
    print("Running current PSO...")
    x_cur, _, _ = particle_swarm(
        objective,
        lb,
        ub,
        n=20,
        max_vel=5.0,
        max_iters=240,
        progress_every=20,
    )
    cur_eval = evaluate_candidate(x_cur)
    results["current"] = {"x": x_cur, "eval": cur_eval, "history": clone_swarm_history(X_history)}
    print(f"Current PSO: obj={cur_eval['objective']:.2f}, length={cur_eval['length']:.2f}, hits={cur_eval['exact_hits']}")
else:
    cur_eval = None

if RUN_PSO_LEGACY:
    print("Running legacy PSO...")
    x_leg, _, _ = particle_swarm_legacy(
        objective,
        lb,
        ub,
        n=20,
        max_vel=5.0,
        max_iters=240,
    )
    leg_eval = evaluate_candidate(x_leg)
    results["legacy"] = {"x": x_leg, "eval": leg_eval, "history": clone_swarm_history(X_history)}
    print(f"Legacy PSO: obj={leg_eval['objective']:.2f}, length={leg_eval['length']:.2f}, hits={leg_eval['exact_hits']}")
else:
    leg_eval = None

if RUN_PSO_6_4:
    print("Running 6_4 PSO...")
    x_64, _, _ = particle_swarm_6_4(
        objective,
        lb,
        ub,
        n=20,
        max_vel=5.0,
    )
    eval_64 = evaluate_candidate(x_64)
    results["6_4"] = {"x": x_64, "eval": eval_64, "history": clone_swarm_history(X_history)}
    print(f"6_4 PSO: obj={eval_64['objective']:.2f}, length={eval_64['length']:.2f}, hits={eval_64['exact_hits']}")
else:
    eval_64 = None

if not results:
    raise RuntimeError("All PSO variants are disabled. Enable at least one RUN_PSO_* toggle.")


def better(eval_a, eval_b):
    if eval_a["exact_hits"] != eval_b["exact_hits"]:
        return eval_a["exact_hits"] < eval_b["exact_hits"]
    return eval_a["objective"] < eval_b["objective"]

# Compare enabled PSOs
best_name = None
best_eval = None
x_pso = None
pso_eval = None
selected_pso_history = []
for name, data in results.items():
    cand_eval = data["eval"]
    if best_eval is None or better(cand_eval, best_eval):
        best_name = name
        best_eval = cand_eval
        x_pso = data["x"]
        pso_eval = cand_eval
        selected_pso_history = data["history"]

print(f"Selected {best_name} PSO output")

def run_ipm_from_seed(seed_x, label):
    print(f"Running interior-point method from {label}...")
    constraints = [
        NonlinearConstraint(
            lambda x: collision_constraint_values(x, BOXES, clearance=COLLISION_CLEARANCE),
            0.0,
            np.inf,
        )
    ]
    history = [np.array(seed_x, dtype=float).copy()]

    def callback(xk, state=None):
        history.append(np.array(xk, dtype=float).copy())

    res_ipm = minimize(
        objective,
        np.array(seed_x, dtype=float),
        method="trust-constr",
        bounds=Bounds(lb, ub),
        constraints=constraints,
        callback=callback,
        options={"maxiter": IPM_MAX_ITERS, "verbose": 0},
    )

    x_ipm = np.array(res_ipm.x, dtype=float)
    ipm_eval = evaluate_candidate(x_ipm)
    buffer_hits, _ = final_collision_audit(ipm_eval["path_curve"], BOXES, margin=IPM_ACCEPT_MARGIN)
    print(f"{label} status: success={res_ipm.success}, message={res_ipm.message}")
    print(f"{label}: obj={ipm_eval['objective']:.2f}, length={ipm_eval['length']:.2f}, hits={ipm_eval['exact_hits']}")
    print(f"{label} margin hits ({IPM_ACCEPT_MARGIN:.1f} m): {buffer_hits}")
    return {
        "x": x_ipm,
        "eval": ipm_eval,
        "history": history,
        "result": res_ipm,
        "margin_hits": buffer_hits,
    }


ipm_rrt_result = None
if RUN_IPM_FROM_RRT:
    ipm_rrt_result = run_ipm_from_seed(x0, "IPM (RRT seed)")
else:
    print("Skipping IPM from RRT seed (RUN_IPM_FROM_RRT=False).")

ipm_pso_result = None
if RUN_IPM_ON_TOP_OF_PSO:
    ipm_pso_result = run_ipm_from_seed(x_pso, "IPM (PSO seed)")
else:
    print("Skipping interior-point refinement from PSO (RUN_IPM_ON_TOP_OF_PSO=False).")

gcs_result = None
if RUN_GCS:
    print("Running GCS planner...")
    gcs_result = run_gcs_planner(plot=False, save_plot=False, save_gif=False)
    print(f"GCS: length={gcs_result['length']:.2f}, hits={gcs_result['exact_hits']}")
else:
    print("Skipping GCS planner (RUN_GCS=False).")

print("Allowed:", 1.0 / RHO)
print("PSO objective:", pso_eval["objective"])
print("PSO length:", pso_eval["length"])
print("PSO max curvature:", pso_eval["max_curvature"])

pso_exact_hits, pso_exact_first = final_collision_audit(pso_eval["path_curve"], BOXES, margin=0.0)
pso_buffer_hits, pso_buffer_first = final_collision_audit(pso_eval["path_curve"], BOXES, margin=IPM_ACCEPT_MARGIN)
print(f"PSO collision check (exact boxes): {'COLLISION' if pso_exact_hits > 0 else 'clear'} | hits={pso_exact_hits}")
if pso_exact_first is not None:
    bi, seg_i, p = pso_exact_first
    print(f"  PSO first exact hit: box={bi}, segment={seg_i}, point=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")
print(f"PSO collision check (with {IPM_ACCEPT_MARGIN:.1f} margin): {'COLLISION' if pso_buffer_hits > 0 else 'clear'} | hits={pso_buffer_hits}")
if pso_buffer_first is not None:
    bi, seg_i, p = pso_buffer_first
    print(f"  PSO first margin hit: box={bi}, segment={seg_i}, point=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")

if ipm_rrt_result is not None:
    ipm_rrt_eval = ipm_rrt_result["eval"]
    print("IPM (RRT seed) objective:", ipm_rrt_eval["objective"])
    print("IPM (RRT seed) length:", ipm_rrt_eval["length"])
    print("IPM (RRT seed) max curvature:", ipm_rrt_eval["max_curvature"])

if ipm_pso_result is not None:
    ipm_pso_eval = ipm_pso_result["eval"]
    print("IPM (PSO seed) objective:", ipm_pso_eval["objective"])
    print("IPM (PSO seed) length:", ipm_pso_eval["length"])
    print("IPM (PSO seed) max curvature:", ipm_pso_eval["max_curvature"])

if gcs_result is not None:
    print("GCS objective (length):", gcs_result["length"])
    print("GCS max curvature:", gcs_result["max_curvature"])

# History summaries for the comparison chart.
pso_length_history, pso_curvature_history, pso_hit_history = history_metrics_from_swarm(selected_pso_history)
ipm_rrt_length_history, ipm_rrt_curvature_history, ipm_rrt_hit_history = ([], [], [])
ipm_pso_length_history, ipm_pso_curvature_history, ipm_pso_hit_history = ([], [], [])
if ipm_rrt_result is not None:
    ipm_rrt_length_history, ipm_rrt_curvature_history, ipm_rrt_hit_history = history_metrics_from_states(ipm_rrt_result["history"])
if ipm_pso_result is not None:
    ipm_pso_length_history, ipm_pso_curvature_history, ipm_pso_hit_history = history_metrics_from_states(ipm_pso_result["history"])

gcs_length_history = gcs_result["length_history"] if gcs_result is not None else []
gcs_curvature_history = gcs_result["curvature_history"] if gcs_result is not None else []
gcs_hit_history = gcs_result["hit_history"] if gcs_result is not None else []

# Use PSO path for traversal visualization while plotting all methods side-by-side.
path_curve = pso_eval["path_curve"]

gif_path = os.path.join(os.path.dirname(__file__), "pathfinder_pso_path_evolution.gif")
if SAVE_GIFS:
    print("Saving PSO path-evolution GIF...")
    visualize_path_evolution(selected_pso_history, objective, gif_path)
    print(f"Saved GIF to {gif_path}")
else:
    print("Skipping PSO path-evolution GIF (SAVE_GIFS=False)")

traverse_gif_path = os.path.join(os.path.dirname(__file__), "pathfinder_pso_agent_traverse.gif")
if SAVE_GIFS:
    print("Saving PSO agent-traverse GIF...")
    visualize_agent_traverse(path_curve, traverse_gif_path)
    print(f"Saved GIF to {traverse_gif_path}")
else:
    print("Skipping PSO agent-traverse GIF (SAVE_GIFS=False)")

# ── Plot optimized results ────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#0d0d1a')

for box in BOXES:
    add_box(ax, box)

ax.plot(pso_eval["x_s"], pso_eval["y_s"], pso_eval["z_s"], color='#ffd166', lw=2.6, label='PSO (selected)')

if PLOT_IPM_FROM_RRT and ipm_rrt_result is not None:
    ax.plot(
        ipm_rrt_result["eval"]["x_s"],
        ipm_rrt_result["eval"]["y_s"],
        ipm_rrt_result["eval"]["z_s"],
        color='#ff8c42',
        lw=2.0,
        linestyle='--',
        label='IPM (RRT seed)',
    )

if PLOT_PSO_IPM and ipm_pso_result is not None:
    ax.plot(
        ipm_pso_result["eval"]["x_s"],
        ipm_pso_result["eval"]["y_s"],
        ipm_pso_result["eval"]["z_s"],
        color='#2ecc71',
        lw=2.0,
        linestyle=':',
        label='PSO + IPM',
    )

if PLOT_GCS and gcs_result is not None:
    ax.plot(
        gcs_result["x_s"],
        gcs_result["y_s"],
        gcs_result["z_s"],
        color='#7f7fff',
        lw=2.0,
        linestyle='-.',
        label='GCS',
    )

ax.plot([START[0]], [START[1]], [START[2]], 'o', color='#2ecc71', ms=10)
ax.plot([GOAL[0]], [GOAL[1]], [GOAL[2]], '*', color='#e74c3c', ms=14)

ax.set_xlim(SPACE_BOUNDS[0], SPACE_BOUNDS[1])
ax.set_ylim(SPACE_BOUNDS[2], SPACE_BOUNDS[3])
ax.set_zlim(SPACE_BOUNDS[4], SPACE_BOUNDS[5])
ax.set_box_aspect((1, 1, 0.55))
ax.view_init(elev=24, azim=-148)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.title("3D B-Spline Paths: PSO, IPM, and GCS")
ax.legend(loc='upper right')
plot_path = os.path.join(os.path.dirname(__file__), "pathfinder_pso_ipm_gcs_3d.png")
plt.savefig(plot_path, dpi=200, bbox_inches='tight')
print(f"Saved plot to {plot_path}")

fig_metrics, (ax_len, ax_curv, ax_hits) = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
fig_metrics.patch.set_facecolor('#0d0d1a')
ax_len.set_facecolor('#0d0d1a')
ax_curv.set_facecolor('#0d0d1a')
ax_hits.set_facecolor('#0d0d1a')

series = [
    ("PSO", pso_length_history, pso_curvature_history, pso_hit_history, '#ffd166'),
    ("IPM (RRT seed)", ipm_rrt_length_history, ipm_rrt_curvature_history, ipm_rrt_hit_history, '#ff8c42'),
    ("PSO + IPM", ipm_pso_length_history, ipm_pso_curvature_history, ipm_pso_hit_history, '#2ecc71'),
    ("GCS", gcs_length_history, gcs_curvature_history, gcs_hit_history, '#7f7fff'),
]

for name, length_hist, curv_hist, hit_hist, color in series:
    if length_hist:
        ax_len.plot(range(len(length_hist)), length_hist, color=color, lw=2.0, label=name)
    if curv_hist:
        ax_curv.plot(range(len(curv_hist)), curv_hist, color=color, lw=2.0, label=name)
    if hit_hist:
        ax_hits.plot(range(len(hit_hist)), hit_hist, color=color, lw=2.0, label=name)

for axis in (ax_len, ax_curv, ax_hits):
    axis.tick_params(colors='white')
    axis.yaxis.label.set_color('white')
    axis.xaxis.label.set_color('white')
    for spine in axis.spines.values():
        spine.set_color('white')
    axis.grid(alpha=0.2, color='white')

ax_len.set_ylabel('Path length')
ax_len.legend(loc='upper right')
ax_len.set_title('Length')

ax_curv.set_ylabel('Max curvature')
ax_curv.legend(loc='upper right')
ax_curv.set_title('Curvature')

ax_hits.set_ylabel('Exact collision hits')
ax_hits.set_xlabel('Iteration / optimization step')
ax_hits.legend(loc='upper right')
ax_hits.set_title('Collision hits')

fig_metrics.suptitle('Path Length, Curvature, and Collision Hits Over Time', color='white')
metrics_path = os.path.join(os.path.dirname(__file__), "pathfinder_path_metrics_history.png")
fig_metrics.savefig(metrics_path, dpi=200, bbox_inches='tight', facecolor=fig_metrics.get_facecolor())
print(f"Saved plot to {metrics_path}")

plt.show()