# Core imports
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import BSpline
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import lhsmdu
import heapq

# ── Configuration ──────────────────────────────────────────────────────────
WORLD_SCALE = 1000.0
RRT_STEP = 0.05 * WORLD_SCALE
RRT_GOAL_RADIUS = 0.08 * WORLD_SCALE
COLLISION_SAMPLE_SPACING = 0.01 * WORLD_SCALE
LOCAL_BOUNDS_MARGIN = 0.10 * WORLD_SCALE
RHO   = 4.0                    # turning radius in meters
N_BOXES = 14
BOX_SEED = 23
SAVE_GIFS = True
COLLISION_CLEARANCE = 0.0
COLLISION_CONSTRAINT_SAMPLES = 600
Z_HEADROOM = 10.0
IPM_MAX_ITERS = 100
IPM_ACCEPT_MARGIN = 0.3

RUN_PSO = True
RUN_IPM_FROM_RRT = True
RUN_IPM_ON_TOP_OF_PSO = True
RUN_GCS = True

# Plot toggles
PLOT_PSO = RUN_PSO
PLOT_IPM_FROM_RRT = RUN_IPM_FROM_RRT
PLOT_PSO_IPM = RUN_IPM_ON_TOP_OF_PSO
PLOT_GCS = RUN_GCS

def boxes_overlap_xy(a, b, gap=0.015):
    return not (
        a[3] + gap <= b[0] or b[3] + gap <= a[0] or
        a[4] + gap <= b[1] or b[4] + gap <= a[1]
    )
# Generate random non-overlapping boxes for obstacles
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

# Lower left open corner (safe margin from edge and obstacles)
START_Z = 5.0
START = (80.0, 120.0, START_Z)
# Upper right open corner (safe margin from edge and obstacles)
GOAL_Z = 0.5 * TALLEST_BOX_Z
GOAL = (920.0, 880.0, GOAL_Z)

def build_bspline(control_pts, degree=3, n_samples=300):
    """Build a B-spline curve from control points."""
    control_pts = np.array(control_pts)

    n = len(control_pts)
    if n < degree + 1:
        raise ValueError(f"Need at least degree + 1 control points, got {n} for degree={degree}.")

    # clamped knot vector
    knots = np.concatenate((
        np.zeros(degree + 1),
        np.linspace(0, 1, n - degree + 1)[1:-1],
        np.ones(degree + 1)
    ))

    t = np.linspace(0, 1, n_samples)

    spline_x = BSpline(knots, control_pts[:,0], degree)
    spline_y = BSpline(knots, control_pts[:,1], degree)
    spline_z = BSpline(knots, control_pts[:,2], degree)

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
    """Compute curvature along a 3D curve."""
    v1 = np.stack([dx, dy, dz], axis=1)
    v2 = np.stack([ddx, ddy, ddz], axis=1)
    num = np.linalg.norm(np.cross(v1, v2), axis=1)
    denom = np.linalg.norm(v1, axis=1)**3 + 1e-6
    return num / denom


def curve_length_from_samples(x_s, y_s, z_s):
    pts = np.column_stack((x_s, y_s, z_s))
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))


# ── Collision helpers ───────────────────────────────────────────────────────

# Check if a point is inside a box (with margin)
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
def rrt(start_xyz, goal_xyz, boxes, n_iter=30000, step=RRT_STEP, goal_r=RRT_GOAL_RADIUS, mg=RHO+0.75):
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
            # Ensure the final connection to the exact goal is also collision free.
            if not seg_clear(newp[0], newp[1], newp[2], goal_arr[0], goal_arr[1], goal_arr[2], boxes, mg):
                continue

            # Trace back including the new node, then append the exact goal point.
            path = []
            idx = len(nodes)-1
            while idx != -1:
                path.append(nodes[idx])
                idx = parent[idx]
            path.reverse()
            path.append(goal_arr)
            return path
    return None

print("Running RRT...")
rrt_path = rrt(START, GOAL, BOXES)
if rrt_path is None:
    raise RuntimeError("RRT failed to find a path — try increasing n_iter")
print(f"  RRT: {len(rrt_path)} nodes")
rrt_xs, rrt_ys, rrt_zs, _, _, _, _, _, _ = build_bspline(rrt_path, n_samples=200)
rrt_full_length = curve_length_from_samples(rrt_xs, rrt_ys, rrt_zs)
print(f"  RRT length: {rrt_full_length:.2f}")

def particle_swarm(f, x_lb, x_ub, seed_x=None, n=20, inertia=0.5, self_influence=1.8, social_influence=1.8, max_vel=5.0, f_tol=1e-6, max_iters=300, stall_iters=25, local_init_frac=0.5):
    global nfev
    global X_history
    nfev = 0
    X_history = []

    if seed_x is None:
        seed_x = np.array(x_lb + 0.5 * (x_ub - x_lb), dtype=float)
    else:
        seed_x = np.array(seed_x, dtype=float)

    dim = len(x_lb)
    lhs_raw = lhsmdu.sample(dim, n)
    lhs_samples = np.array(lhs_raw).T
    # Start with globally distributed particles over the full provided bounds.
    X = x_lb + (x_ub - x_lb) * lhs_samples
    X[0, :] = seed_x.copy()

    # Keep a subset of particles near the RRT seed for exploitation.
    n_local = int(max(1, min(n - 1, round((n - 1) * local_init_frac))))
    if n_local > 0:
        sigma = 0.08 * (x_ub - x_lb)
        local_noise = np.random.randn(n_local, dim) * sigma
        X[1:1 + n_local, :] = seed_x[None, :] + local_noise
    X = np.maximum(np.minimum(X, x_ub), x_lb)
    dX = (np.random.rand(n, len(x_lb)) * 2 - 1) * max_vel
    X_best = X.copy()
    best_evals = [evaluate_candidate(np.array(xi, dtype=float), objective_fn=f) for xi in X_best]
    nfev += len(X_best)

    feasible_best_x = None
    feasible_best_eval = None
    for xi, eval_data in zip(X_best, best_evals):
        if int(eval_data["exact_hits"]) == 0:
            if feasible_best_eval is None or score_candidate_eval(eval_data) < score_candidate_eval(feasible_best_eval):
                feasible_best_x = np.array(xi, dtype=float).copy()
                feasible_best_eval = eval_data

    n_conv = 0
    best_score_prev = None
    converged = False

    for _ in range(max_iters):
        current_evals = [evaluate_candidate(np.array(xi, dtype=float), objective_fn=f) for xi in X]
        nfev += len(X)
        current_scores = [score_candidate_eval(eval_data) for eval_data in current_evals]
        # Global-best should come from personal-best memory, not only current positions.
        pbest_scores = [score_candidate_eval(eval_data) for eval_data in best_evals]
        feasible_indices = [idx for idx, eval_data in enumerate(best_evals) if int(eval_data["exact_hits"]) == 0]
        if feasible_indices:
            swarm_best_i = min(feasible_indices, key=lambda idx: pbest_scores[idx])
        else:
            swarm_best_i = min(range(n), key=lambda idx: pbest_scores[idx])

        for i in range(n):
            if current_scores[i] < score_candidate_eval(best_evals[i]):
                X_best[i, :] = X[i, :]
                best_evals[i] = current_evals[i]

            if int(current_evals[i]["exact_hits"]) == 0:
                if feasible_best_eval is None or score_candidate_eval(current_evals[i]) < score_candidate_eval(feasible_best_eval):
                    feasible_best_x = X[i, :].copy()
                    feasible_best_eval = current_evals[i]

        swarm_best = X[swarm_best_i, :].copy()

        for i in range(n):
            dX[i, :] = inertia * dX[i, :] + self_influence * (X_best[i, :] - X[i, :]) + social_influence * (swarm_best - X[i, :])
            dX[i, :] = np.maximum(np.minimum(dX[i, :], max_vel), -max_vel)
            X[i, :] += dX[i, :]
            X[i, :] = np.maximum(np.minimum(X[i, :], x_ub), x_lb)

        best_index = min(range(n), key=lambda idx: score_candidate_eval(best_evals[idx]))
        best_score = float(best_evals[best_index]["objective"])
        if best_score_prev is not None:
            # Stop only after a sustained stall in relative improvement.
            improvement = best_score_prev - best_score
            rel_tol = f_tol * max(1.0, abs(best_score_prev))
            if improvement <= rel_tol:
                n_conv += 1
                if n_conv >= stall_iters:
                    converged = True
                    break
            else:
                n_conv = 0
        best_score_prev = best_score
        X_history.append(X.copy())
        max_vel *= 0.9

    if not converged:
        print("PSO reached max_iters; returning best particle found.")

    if feasible_best_x is not None:
        return feasible_best_x, float(feasible_best_eval["objective"]), nfev

    best_index = min(range(n), key=lambda idx: score_candidate_eval(best_evals[idx]))
    r_x = X_best[best_index]
    return r_x, float(best_evals[best_index]["objective"]), nfev


def evaluate_candidate(x, objective_fn=None):
    if objective_fn is None:
        objective_fn = objective
    x_s, y_s, z_s, length, curvature, _ = build_spline_from_x(x)
    path_curve = np.column_stack((x_s, y_s, z_s))
    exact_hits, _ = final_collision_audit(path_curve, BOXES, margin=0.0)
    margin_hits, _ = final_collision_audit(path_curve, BOXES, margin=IPM_ACCEPT_MARGIN)
    max_curvature = float(np.max(curvature))
    return {
        "objective": float(objective_fn(x)),
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


def evaluate_waypoint_path(path_xyz, n_samples=200):
    """Evaluate a waypoint path with the same spline-sampled metrics used by optimization candidates."""
    x_s, y_s, z_s, dx, dy, dz, ddx, ddy, ddz = build_bspline(path_xyz, n_samples=n_samples)
    curvature = compute_curvature(dx, dy, dz, ddx, ddy, ddz)
    length = curve_length_from_samples(x_s, y_s, z_s)
    path_curve = np.column_stack((x_s, y_s, z_s))
    exact_hits, _ = final_collision_audit(path_curve, BOXES, margin=0.0)
    return {
        "length": float(length),
        "max_curvature": float(np.max(curvature)),
        "exact_hits": int(exact_hits),
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



def visualize_agents_multi(paths_dict, output_path, max_frames=120, fps=64):
    """Animate agents at a shared speed so shorter paths finish earlier."""
    # paths_dict: {label: (N,3) array}
    raw_paths = {}
    path_lengths = {}
    max_path_length = 0.0

    for label, path_xyz in paths_dict.items():
        path_xyz = np.asarray(path_xyz, dtype=float)
        raw_paths[label] = path_xyz
        if len(path_xyz) < 2:
            path_lengths[label] = 0.0
            continue
        seg_lengths = np.linalg.norm(path_xyz[1:] - path_xyz[:-1], axis=1)
        total_length = float(np.sum(seg_lengths))
        path_lengths[label] = total_length
        max_path_length = max(max_path_length, total_length)

    if max_path_length <= 1e-9:
        return

    # One global distance step per frame: all agents travel at the same speed.
    distance_per_frame = max_path_length / max(1, max_frames - 1)
    max_len = max_frames
    path_pts = {}

    for label, path_xyz in raw_paths.items():
        total_length = path_lengths[label]
        if len(path_xyz) < 2 or total_length <= 1e-9:
            anchor = path_xyz[-1] if len(path_xyz) else np.array(START, dtype=float)
            path_pts[label] = np.repeat(anchor[None, :], max_len, axis=0)
            continue

        seg_vec = path_xyz[1:] - path_xyz[:-1]
        seg_lengths = np.linalg.norm(seg_vec, axis=1)
        cum_dist = np.concatenate(([0.0], np.cumsum(seg_lengths)))

        n_steps = int(np.ceil(total_length / distance_per_frame))
        d_samples = np.linspace(0.0, total_length, n_steps + 1)
        pts = []
        for d in d_samples:
            j = int(np.searchsorted(cum_dist, d, side='right') - 1)
            j = min(max(j, 0), len(seg_lengths) - 1)
            seg_len = max(seg_lengths[j], 1e-12)
            t = (d - cum_dist[j]) / seg_len
            t = min(max(t, 0.0), 1.0)
            pts.append((1.0 - t) * path_xyz[j] + t * path_xyz[j + 1])

        pts = np.asarray(pts, dtype=float)
        if len(pts) < max_len:
            pad = np.repeat(pts[-1][None, :], max_len - len(pts), axis=0)
            pts = np.concatenate([pts, pad], axis=0)
        else:
            pts = pts[:max_len]
        path_pts[label] = pts

    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0d0d1a')
    for box in BOXES:
        add_box(ax, box, alpha=0.22)
    # Plot all full paths
    colors = {
        'RRT': '#cfd3d6',
        'PSO': '#ffd166',
        'IPM (RRT seed)': '#ff8c42',
        'PSO + IPM': '#2ecc71',
        'GCS': '#7f7fff',
    }
    for label, pts in path_pts.items():
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=colors.get(label, None), lw=2.1, alpha=0.45)
    ax.plot([START[0]], [START[1]], [START[2]], 'o', color='#2ecc71', ms=10)
    ax.plot([GOAL[0]], [GOAL[1]], [GOAL[2]], '*', color='#e74c3c', ms=14)
    # One agent per path
    agents = {}
    trails = {}
    legend_handles = []
    for label, pts in path_pts.items():
        agents[label] = ax.scatter([], [], [], s=130, c=colors.get(label, '#ffdd57'), edgecolors='black', linewidths=1.0, depthshade=False, label=label)
        trails[label], = ax.plot([], [], [], color=colors.get(label, '#00d1ff'), lw=3.0)
        legend_handles.append(Line2D([0], [0], color=colors.get(label, '#00d1ff'), lw=3.0, label=label))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect((1, 1, 0.55))
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.85)

    # Keep a fixed frame for the full animation so the whole trajectory remains visible.
    ax.set_xlim(SPACE_BOUNDS[0], SPACE_BOUNDS[1])
    ax.set_ylim(SPACE_BOUNDS[2], SPACE_BOUNDS[3])
    ax.set_zlim(SPACE_BOUNDS[4], SPACE_BOUNDS[5])
    ax.view_init(elev=32, azim=-148)

    def update(frame_i):
        for label, pts in path_pts.items():
            pos = pts[frame_i]
            history = pts[: frame_i + 1]
            agents[label]._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
            trails[label].set_data(history[:, 0], history[:, 1])
            trails[label].set_3d_properties(history[:, 2])
        ax.set_title(f"Agent Traverse | frame {frame_i + 1}/{max_len}")
        return list(agents.values()) + list(trails.values())

    anim = FuncAnimation(fig, update, frames=max_len, interval=int(1000 / fps), blit=False)
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
    length = curve_length_from_samples(x_s, y_s, z_s)

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
    collision_penalty = 1e9 if (hits > 0 or collision) else 0

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



# Deep copy of swarm history (for animation)
def clone_swarm_history(history):
    return [np.array(frame, dtype=float).copy() for frame in history]



# Helper for sorting candidate evaluations
def score_candidate_eval(eval_data):
    return (float(eval_data["objective"]),)


def is_feasible_eval(eval_data, curvature_tol=1e-4):
    return (
        int(eval_data["exact_hits"]) == 0
        and float(eval_data.get("max_curvature_violation", float("inf"))) <= curvature_tol
    )



# Extract metrics from swarm history for plotting
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



# Extract metrics from state history for plotting
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


def dijkstra_weighted(cells, neighbors, start_idx, goal_idx, edge_cost_fn):
    dist = {start_idx: 0.0}
    prev = {}
    pq = [(0.0, start_idx)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == goal_idx:
            break
        if d > dist.get(u, float("inf")):
            continue
        for v in neighbors[u]:
            w = float(edge_cost_fn(u, v))
            if not np.isfinite(w) or w <= 0.0:
                continue
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


def build_cell_clearances(cells, boxes):
    clearances = []
    for cell in cells:
        c = cell_center(cell)
        min_dist = min(point_box_signed_distance(c, box) for box in boxes)
        clearances.append(float(min_dist))
    return np.asarray(clearances, dtype=float)


def generate_candidate_cell_paths(cells, neighbors, start_idx, goal_idx, boxes, n_random=8, seed=31):
    """Generate diverse corridor candidates and return unique cell-index paths."""
    centers = [cell_center(c) for c in cells]
    clearances = build_cell_clearances(cells, boxes)
    rng = np.random.default_rng(seed)
    unique = []
    seen = set()

    def add_path(path):
        if path is None:
            return
        key = tuple(path)
        if key in seen:
            return
        seen.add(key)
        unique.append(path)

    def make_cost_fn(clearance_bias=0.0, jitter=0.0):
        def cost(u, v):
            base = float(np.linalg.norm(centers[v] - centers[u]))
            clear = max(1e-3, min(clearances[u], clearances[v]))
            penalty = clearance_bias / clear
            if jitter > 0.0:
                noise = 1.0 + jitter * (rng.random() - 0.5)
                noise = max(0.7, noise)
            else:
                noise = 1.0
            return (base + penalty) * noise
        return cost

    # Deterministic baselines from shortest to increasingly clearance-biased corridors.
    add_path(dijkstra(cells, neighbors, start_idx, goal_idx))
    for clearance_bias in (25.0, 60.0, 120.0):
        add_path(dijkstra_weighted(cells, neighbors, start_idx, goal_idx, make_cost_fn(clearance_bias=clearance_bias, jitter=0.0)))

    # Randomized weighted solves to explore alternate corridors.
    for _ in range(n_random):
        bias = float(rng.uniform(20.0, 130.0))
        jitter = float(rng.uniform(0.12, 0.35))
        add_path(dijkstra_weighted(cells, neighbors, start_idx, goal_idx, make_cost_fn(clearance_bias=bias, jitter=jitter)))

    return unique

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
    max_optimized_portals = 20
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
        x, y, z, dx, dy, dz, ddx, ddy, ddz = build_bspline(pts)
        length = curve_length_from_samples(x, y, z)
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

    def curvature_constraint_flat(flat):
        wps = unpack(flat)
        pts = [np.array(start_xyz, dtype=float), *wps, np.array(goal_xyz, dtype=float)]
        _, _, _, dx, dy, dz, ddx, ddy, ddz = build_bspline(pts, n_samples=COLLISION_CONSTRAINT_SAMPLES)
        curvature = compute_curvature(dx, dy, dz, ddx, ddy, ddz)
        return (0.8 * (1.0 / RHO)) - curvature

    def collision_constraint_flat(flat):
        wps = unpack(flat)
        pts = [np.array(start_xyz, dtype=float), *wps, np.array(goal_xyz, dtype=float)]
        x_s, y_s, z_s, _, _, _, _, _, _ = build_bspline(pts, n_samples=COLLISION_CONSTRAINT_SAMPLES)
        total = len(x_s)
        if total == 0:
            return np.array([], dtype=float)
        indices = np.linspace(0, total - 1, max(2, min(COLLISION_CONSTRAINT_SAMPLES, total)), dtype=int)
        values = []
        for idx in indices:
            p = (x_s[idx], y_s[idx], z_s[idx])
            min_dist = min(point_box_signed_distance(p, box) for box in BOXES)
            values.append(min_dist - COLLISION_CLEARANCE)
        return np.asarray(values, dtype=float)
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
        method="SLSQP",
        bounds=bounds,
        constraints=[
            {"type": "ineq", "fun": collision_constraint_flat},
            {"type": "ineq", "fun": curvature_constraint_flat},
        ],
        callback=callback,
        options={"maxiter": 350, "ftol": 1e-6, "disp": False},
    )
    if res.success:
        center = np.array(res.x, dtype=float)
        best_valid = None
        best_valid_metrics = None
        best_invalid = None
        best_invalid_metrics = None
        kappa_max = 1.0 / RHO
        for base_x in [np.array(res.x, dtype=float), center]:
            for radius_scale in [0.20, 0.16, 0.12, 0.08]:
                local_bounds = []
                for idx, (lo, hi) in enumerate(bounds):
                    span = hi - lo
                    radius = radius_scale * span
                    local_bounds.append((max(lo, base_x[idx] - radius), min(hi, base_x[idx] + radius)))
                res2 = minimize(
                    objective,
                    base_x,
                    method="SLSQP",
                    bounds=local_bounds,
                    constraints=[
                        {"type": "ineq", "fun": collision_constraint_flat},
                        {"type": "ineq", "fun": curvature_constraint_flat},
                    ],
                    callback=callback,
                    options={"maxiter": 220, "ftol": 1e-7, "disp": False},
                )
                if not res2.success:
                    continue
                res2_metrics = evaluate_flat(np.array(res2.x, dtype=float))
                res2_curv = float(np.max(res2_metrics["curvature"]))
                if res2_curv <= kappa_max:
                    if best_valid_metrics is None or res2_metrics["objective"] < best_valid_metrics["objective"]:
                        best_valid = res2
                        best_valid_metrics = res2_metrics
                elif best_invalid_metrics is None or res2_metrics["objective"] < best_invalid_metrics["objective"]:
                    best_invalid = res2
                    best_invalid_metrics = res2_metrics

        if best_valid is not None:
            res = best_valid
        elif best_invalid is not None and best_invalid.fun <= res.fun:
            res = best_invalid
    if not res.success:
        return (mids, history) if return_history else mids
    result = unpack(res.x)
    final_metrics = evaluate_flat(np.array(res.x, dtype=float))
    kappa_max = 1.0 / RHO
    if float(np.max(final_metrics["curvature"])) > kappa_max:
        print(f"WARNING: GCS final path violates curvature ({float(np.max(final_metrics['curvature'])):.6f} > {kappa_max:.6f}). Returning fallback midpoint path.")
        return (mids, history) if return_history else mids
    if not history["length"] or abs(history["length"][-1] - final_metrics["length"]) > 1e-9:
        history["objective"].append(final_metrics["objective"])
        history["length"].append(final_metrics["length"])
        history["curvature"].append(float(np.max(final_metrics["curvature"])))
        history["hits"].append(int(final_metrics["hits"]))
    return (result, history) if return_history else result

def run_gcs_planner(plot=True, save_plot=True, save_gif=False):
    start = np.array(START, dtype=float)
    goal = np.array(GOAL, dtype=float)
    print(f"Start: {tuple(start)}")
    print(f"Goal:  {tuple(goal)}")
    print("Running direct GCS trajectory optimization with curvature in objective...")

    n_internal_waypoints = 14
    t_vals = np.linspace(0.0, 1.0, n_internal_waypoints + 2)[1:-1]
    z_peak = min(SPACE_BOUNDS[5] - 0.5, max(start[2], goal[2], TALLEST_BOX_Z + 2.0))

    init_wps = []
    for t in t_vals:
        p = (1.0 - t) * start + t * goal
        z_line = p[2]
        p[2] = z_line + (z_peak - z_line) * np.sin(np.pi * t)
        init_wps.append(p)
    init_wps = np.asarray(init_wps, dtype=float)

    x0 = init_wps.reshape(-1)
    lb = np.tile(np.array([SPACE_BOUNDS[0], SPACE_BOUNDS[2], SPACE_BOUNDS[4]], dtype=float), n_internal_waypoints)
    ub = np.tile(np.array([SPACE_BOUNDS[1], SPACE_BOUNDS[3], SPACE_BOUNDS[5]], dtype=float), n_internal_waypoints)

    kappa_max = 1.0 / RHO
    gcs_constraint_margin = 0.3
    history = {"objective": [], "length": [], "curvature": [], "hits": []}

    def unpack(flat):
        arr = np.array(flat, dtype=float).reshape(-1, 3)
        return [arr[i] for i in range(arr.shape[0])]

    def evaluate_flat(flat, n_samples=300):
        wps = unpack(flat)
        pts = [start, *wps, goal]
        x_s, y_s, z_s, dx, dy, dz, ddx, ddy, ddz = build_bspline(pts, n_samples=n_samples)
        length = curve_length_from_samples(x_s, y_s, z_s)
        curvature = compute_curvature(dx, dy, dz, ddx, ddy, ddz)
        curvature_violation = np.maximum(0.0, curvature - kappa_max)
        curvature_penalty = float(np.sum(curvature_violation ** 2) * 1e4)
        path_curve = np.column_stack((x_s, y_s, z_s))
        hits, _ = final_collision_audit_gcs(path_curve, BOXES, margin=gcs_constraint_margin)
        return {
            "pts": pts,
            "x_s": x_s,
            "y_s": y_s,
            "z_s": z_s,
            "length": float(length),
            "curvature": curvature,
            "max_curvature": float(np.max(curvature)),
            "objective": float(length + curvature_penalty),
            "hits": int(hits),
            "path_curve": path_curve,
        }

    def objective(flat):
        return evaluate_flat(flat)["objective"]

    def collision_constraint_flat(flat):
        eval_data = evaluate_flat(flat, n_samples=COLLISION_CONSTRAINT_SAMPLES)
        x_s = eval_data["x_s"]
        y_s = eval_data["y_s"]
        z_s = eval_data["z_s"]
        values = []
        for px, py, pz in zip(x_s, y_s, z_s):
            p = (float(px), float(py), float(pz))
            min_dist = min(point_box_signed_distance(p, box) for box in BOXES)
            values.append(min_dist - gcs_constraint_margin)
        return np.asarray(values, dtype=float)

    def curvature_constraint_flat(flat):
        eval_data = evaluate_flat(flat, n_samples=COLLISION_CONSTRAINT_SAMPLES)
        return kappa_max - eval_data["curvature"]

    def callback(flat):
        metrics = evaluate_flat(flat)
        history["objective"].append(metrics["objective"])
        history["length"].append(metrics["length"])
        history["curvature"].append(metrics["max_curvature"])
        history["hits"].append(metrics["hits"])

    initial_metrics = evaluate_flat(x0)
    history["objective"].append(initial_metrics["objective"])
    history["length"].append(initial_metrics["length"])
    history["curvature"].append(initial_metrics["max_curvature"])
    history["hits"].append(initial_metrics["hits"])

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=list(zip(lb, ub)),
        constraints=[
            {"type": "ineq", "fun": collision_constraint_flat},
            {"type": "ineq", "fun": curvature_constraint_flat},
        ],
        callback=callback,
        options={"maxiter": 450, "ftol": 1e-7, "disp": False},
    )

    x_best = np.array(res.x if res.success else x0, dtype=float)
    final_metrics = evaluate_flat(x_best)
    path_xyz = final_metrics["pts"]
    x_s = final_metrics["x_s"]
    y_s = final_metrics["y_s"]
    z_s = final_metrics["z_s"]
    length = final_metrics["length"]
    curvature = final_metrics["curvature"]
    max_kappa = final_metrics["max_curvature"]
    path_curve = final_metrics["path_curve"]
    clear = all(seg_clear_gcs(path_xyz[i], path_xyz[i + 1], BOXES) for i in range(len(path_xyz) - 1))
    z_min = float(np.min(z_s))
    z_max = float(np.max(z_s))

    print(f"Optimizer success: {res.success} | message: {res.message}")
    print(f"Length: {length:.2f}")
    print(f"Collision free: {clear}")
    print(f"Max curvature: {max_kappa:.6f}")
    print(f"Allowed: {kappa_max:.6f}")
    print(f"Collision constraint margin: {gcs_constraint_margin:.1f} m")
    if max_kappa > kappa_max:
        print(f"WARNING: GCS final path violates curvature constraint! {max_kappa:.6f} > {kappa_max:.6f}")
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
        # Default view: above and behind the start corner
        ax.view_init(elev=24, azim=-58)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title(f"3D GCS Path | Length={length:.1f} | max k={float(np.max(curvature)):.3f}")
        if save_plot:
            plt.savefig(plot_path, dpi=200, bbox_inches="tight")
            print(f"Saved plot to {plot_path}")

        # Save top-down view
        ax.view_init(elev=90, azim=-90)
        topdown_path = os.path.join(os.path.dirname(__file__), "gcs_topdown.png")
        plt.savefig(topdown_path, dpi=200, bbox_inches="tight")
        print(f"Saved top-down view to {topdown_path}")

        # Save above and behind start corner view
        ax.view_init(elev=32, azim=-148)
        behind_path = os.path.join(os.path.dirname(__file__), "gcs_behind_start.png")
        plt.savefig(behind_path, dpi=200, bbox_inches="tight")
        print(f"Saved above/behind start view to {behind_path}")

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
        "length_history": list(history["length"]) if history["length"] and any(abs(v) > 1e-9 for v in history["length"]) else [float(length)],
        "curvature_history": list(history["curvature"]),
        "hit_history": list(history["hits"]),
        "objective_history": list(history["objective"]),
        "plot_path": plot_path,
    }

# ── Run PSO with RRT-seeded feasible initialization ────────────────────────
seed_name = "full RRT"
opt_nodes = rrt_path[1:-1]
lb = None
ub = None
x0 = None

if len(opt_nodes) > 0:
    lb_c, ub_c, x0_c = build_bounds_and_initial_guess(opt_nodes, margin)
    if is_collision_free_solution(x0_c):
        lb, ub, x0 = lb_c, ub_c, x0_c

if x0 is None:
    raise RuntimeError("No feasible RRT-seeded initial solution found. Try increasing RRT n_iter or LOCAL_BOUNDS_MARGIN.")

print(f"Optimization dim: {len(lb)}")
print(f"Using feasible initial seed from {seed_name} with {len(opt_nodes)} waypoints")

results = {}

if RUN_PSO:
    print("Running PSO...")
    seed_eval = evaluate_candidate(x0)
    print(f"  Seed collision check: exact_hits={seed_eval['exact_hits']}, objective={seed_eval['objective']:.2f}")
    x_leg, _, _ = particle_swarm(
        objective,
        np.tile(np.array([SPACE_BOUNDS[0], SPACE_BOUNDS[2], SPACE_BOUNDS[4]], dtype=float), len(opt_nodes)),
        np.tile(np.array([SPACE_BOUNDS[1], SPACE_BOUNDS[3], SPACE_BOUNDS[5]], dtype=float), len(opt_nodes)),
        seed_x=x0,
        n=56,
        max_vel=12.0,
        max_iters=240,
        local_init_frac=0.45,
    )
    leg_eval = evaluate_candidate(x_leg)
    results["PSO"] = {"x": x_leg, "eval": leg_eval, "history": clone_swarm_history(X_history)}
    print(f"PSO: obj={leg_eval['objective']:.2f}, length={leg_eval['length']:.2f}, hits={leg_eval['exact_hits']}")
else:
    leg_eval = None

if not results:
    print("PSO is disabled. Enable RUN_PSO toggle.")

def better(eval_a, eval_b):
    if eval_a["exact_hits"] != eval_b["exact_hits"]:
        return eval_a["exact_hits"] < eval_b["exact_hits"]
    return eval_a["objective"] < eval_b["objective"]


def curvature_constraint(x):
    pts = [START[:3]]
    for i in range(0, len(x), 3):
        pts.append((x[i], x[i + 1], x[i + 2]))
    pts.append(GOAL[:3])
    _, _, _, dx, dy, dz, ddx, ddy, ddz = build_bspline(pts, n_samples=COLLISION_CONSTRAINT_SAMPLES)
    curvature = compute_curvature(dx, dy, dz, ddx, ddy, ddz)
    kappa_max = 1.0 / RHO
    return kappa_max - curvature


def cylindrical_boundary_constraint(x, boxes=BOXES, n_checks=COLLISION_CONSTRAINT_SAMPLES):
    """
    Enforce that all path points are outside a cylinder (in x-y) around each box.
    Cylinder center: box x-y center; radius: half the x-y diagonal of the box.
    Returns: array of min distance from each path sample to the surface of all cylinders (should be >= 0).
    """
    x_s, y_s, z_s, _, _, _ = build_spline_from_x(x, n_samples=n_checks)
    total = len(x_s)
    if total == 0:
        return np.array([], dtype=float)

    n_checks = max(2, min(int(n_checks), total))
    indices = np.linspace(0, total - 1, n_checks, dtype=int)

    values = []
    for idx in indices:
        px, py = x_s[idx], y_s[idx]
        # For each box, compute cylinder center and radius
        min_dist = float('inf')
        for box in boxes:
            x0, y0, _, x1, y1, _ = box
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            dx = x1 - x0
            dy = y1 - y0
            radius = 0.5 * np.sqrt(dx**2 + dy**2)
            dist = np.sqrt((px - cx)**2 + (py - cy)**2) - radius
            if dist < min_dist:
                min_dist = dist
        values.append(min_dist)
    return np.asarray(values, dtype=float)

def run_ipm_from_seed(seed_x, label):
    print(f"Running interior-point method from {label}...")
    constraints = [
        NonlinearConstraint(
            cylindrical_boundary_constraint,
            0.0,
            np.inf,
        ),
        NonlinearConstraint(
            curvature_constraint,
            0.0,
            np.inf,
        ),
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
    ipm_pso_result = run_ipm_from_seed(x_leg, "IPM (PSO seed)")
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
if leg_eval is not None:
    print("PSO objective:", leg_eval["objective"])
    print("PSO length:", leg_eval["length"])
    print("PSO max curvature:", leg_eval["max_curvature"])

    pso_exact_hits, pso_exact_first = final_collision_audit(leg_eval["path_curve"], BOXES, margin=0.0)
    pso_buffer_hits, pso_buffer_first = final_collision_audit(leg_eval["path_curve"], BOXES, margin=IPM_ACCEPT_MARGIN)
    print(f"PSO collision check (exact boxes): {'COLLISION' if pso_exact_hits > 0 else 'clear'} | hits={pso_exact_hits}")
    if pso_exact_first is not None:
        bi, seg_i, p = pso_exact_first
        print(f"  PSO first exact hit: box={bi}, segment={seg_i}, point=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")
    print(f"PSO collision check (with {IPM_ACCEPT_MARGIN:.1f} margin): {'COLLISION' if pso_buffer_hits > 0 else 'clear'} | hits={pso_buffer_hits}")
    if pso_buffer_first is not None:
        bi, seg_i, p = pso_buffer_first
        print(f"  PSO first margin hit: box={bi}, segment={seg_i}, point=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")
else:
    print("PSO disabled; skipping PSO metrics.")

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


# Define selected_pso_history and pso_eval for plotting and metrics
selected_pso_history = results["PSO"]["history"] if "PSO" in results and "history" in results["PSO"] else []
pso_eval = leg_eval

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

# RRT baseline metrics for history plots (constant reference lines),
# evaluated with the same spline-sampled representation used elsewhere.
rrt_eval = evaluate_waypoint_path(rrt_path, n_samples=200)
rrt_length = rrt_eval["length"]
rrt_max_curvature = rrt_eval["max_curvature"]
rrt_plot_curve = np.asarray(rrt_eval["path_curve"], dtype=float)
metrics_horizon = max(
    1,
    len(pso_length_history),
    len(ipm_rrt_length_history),
    len(ipm_pso_length_history),
    len(gcs_length_history),
)

def align_history(history, horizon):
    if horizon <= 0:
        return []
    if not history:
        return [np.nan] * horizon
    aligned = list(history)
    if len(aligned) < horizon:
        # Pad with NaN so the plotted line ends at the method's last iteration.
        aligned.extend([np.nan] * (horizon - len(aligned)))
    return aligned[:horizon]

pso_length_history = align_history(pso_length_history, metrics_horizon)
pso_curvature_history = align_history(pso_curvature_history, metrics_horizon)
ipm_rrt_length_history = align_history(ipm_rrt_length_history, metrics_horizon)
ipm_rrt_curvature_history = align_history(ipm_rrt_curvature_history, metrics_horizon)
ipm_pso_length_history = align_history(ipm_pso_length_history, metrics_horizon)
ipm_pso_curvature_history = align_history(ipm_pso_curvature_history, metrics_horizon)
gcs_length_history = align_history(gcs_length_history, metrics_horizon)
gcs_curvature_history = align_history(gcs_curvature_history, metrics_horizon)

rrt_length_history = [rrt_length] * metrics_horizon
rrt_curvature_history = [rrt_max_curvature] * metrics_horizon

# Use PSO path for traversal visualization while plotting all methods side-by-side.
if pso_eval is not None:
    path_curve = pso_eval["path_curve"]
elif gcs_result is not None:
    path_curve = gcs_result["path_curve"]
else:
    path_curve = np.asarray(rrt_path, dtype=float)


gif_path = os.path.join(os.path.dirname(__file__), "pathfinder_pso_path_evolution.gif")
if SAVE_GIFS:
    print("Saving PSO path-evolution GIF...")
    visualize_path_evolution(selected_pso_history, objective, gif_path)
    print(f"Saved GIF to {gif_path}")
else:
    print("Skipping PSO path-evolution GIF (SAVE_GIFS=False)")

traverse_gif_path = os.path.join(os.path.dirname(__file__), "pathfinder_agents_traverse.gif")
if SAVE_GIFS:
    print("Saving multi-agent traverse GIF...")
    paths_dict = {}
    paths_dict['RRT'] = rrt_plot_curve
    if pso_eval is not None:
        paths_dict['PSO'] = pso_eval["path_curve"]
    if PLOT_IPM_FROM_RRT and ipm_rrt_result is not None:
        paths_dict['IPM (RRT seed)'] = ipm_rrt_result["eval"]["path_curve"]
    if PLOT_PSO_IPM and ipm_pso_result is not None:
        paths_dict['PSO + IPM'] = ipm_pso_result["eval"]["path_curve"]
    if PLOT_GCS and gcs_result is not None:
        paths_dict['GCS'] = gcs_result["path_curve"]
    visualize_agents_multi(paths_dict, traverse_gif_path)
    print(f"Saved GIF to {traverse_gif_path}")
else:
    print("Skipping agent traverse GIF (SAVE_GIFS=False)")

# ── Plot optimized results ────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#0d0d1a')

for box in BOXES:
    add_box(ax, box)


# Plot full RRT path
rrt_arr = rrt_plot_curve
ax.plot(rrt_arr[:,0], rrt_arr[:,1], rrt_arr[:,2], color='#9aa3ad', lw=2.8, linestyle='--', label='RRT', zorder=21)

if pso_eval is not None:
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

fig_birds_eye = plt.figure(figsize=(11, 9))
ax_be = fig_birds_eye.add_subplot(111)
ax_be.set_facecolor('#0d0d1a')
for box in BOXES:
    x0, y0, _, x1, y1, _ = box
    ax_be.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor='#922b21', edgecolor='#e74c3c', alpha=0.35, linewidth=0.8))

topdown_series = [
    ('RRT', rrt_arr, '#9aa3ad', '--'),
]
if pso_eval is not None:
    topdown_series.append(('PSO', np.column_stack((pso_eval['x_s'], pso_eval['y_s'])), '#ffd166', '-'))
if PLOT_IPM_FROM_RRT and ipm_rrt_result is not None:
    topdown_series.append(('IPM (RRT seed)', np.column_stack((ipm_rrt_result['eval']['x_s'], ipm_rrt_result['eval']['y_s'])), '#ff8c42', '--'))
if PLOT_PSO_IPM and ipm_pso_result is not None:
    topdown_series.append(('PSO + IPM', np.column_stack((ipm_pso_result['eval']['x_s'], ipm_pso_result['eval']['y_s'])), '#2ecc71', ':'))
if PLOT_GCS and gcs_result is not None:
    topdown_series.append(('GCS', np.column_stack((gcs_result['x_s'], gcs_result['y_s'])), '#7f7fff', '-.'))

for label, pts_xy, color, linestyle in topdown_series:
    ax_be.plot(pts_xy[:, 0], pts_xy[:, 1], color=color, lw=2.5, linestyle=linestyle, label=label)

ax_be.scatter([START[0]], [START[1]], color='#2ecc71', s=70, zorder=10, label='Start')
ax_be.scatter([GOAL[0]], [GOAL[1]], color='#e74c3c', s=90, marker='*', zorder=10, label='Goal')
ax_be.set_xlim(SPACE_BOUNDS[0], SPACE_BOUNDS[1])
ax_be.set_ylim(SPACE_BOUNDS[2], SPACE_BOUNDS[3])
ax_be.set_aspect('equal', adjustable='box')
ax_be.set_xlabel('X')
ax_be.set_ylabel('Y')
ax_be.set_title('Birds-Eye View of Trajectories')
ax_be.legend(loc='upper right')
birds_eye_path = os.path.join(os.path.dirname(__file__), "pathfinder_pso_ipm_gcs_birds_eye.png")
fig_birds_eye.savefig(birds_eye_path, dpi=200, bbox_inches='tight', facecolor=fig_birds_eye.get_facecolor())
print(f"Saved plot to {birds_eye_path}")

fig_metrics, (ax_len, ax_curv) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
fig_metrics.patch.set_facecolor('#0d0d1a')
ax_len.set_facecolor('#0d0d1a')
ax_curv.set_facecolor('#0d0d1a')

series = [
    ("RRT", rrt_length_history, rrt_curvature_history, '#9aa3ad', '--'),
    ("PSO", pso_length_history, pso_curvature_history, '#ffd166', '-'),
    ("IPM (RRT seed)", ipm_rrt_length_history, ipm_rrt_curvature_history, '#ff8c42', '--'),
    ("PSO + IPM", ipm_pso_length_history, ipm_pso_curvature_history, '#2ecc71', ':'),
    ("GCS", gcs_length_history, gcs_curvature_history, '#7f7fff', '-.'),
]

for name, length_hist, curv_hist, color, linestyle in series:
    if length_hist:
        ax_len.plot(range(len(length_hist)), length_hist, color=color, linestyle=linestyle, lw=2.0, label=name)
    if curv_hist:
        ax_curv.plot(range(len(curv_hist)), curv_hist, color=color, linestyle=linestyle, lw=2.0, label=name)

for axis in (ax_len, ax_curv):
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
ax_curv.set_xlabel('Iteration / optimization step')

fig_metrics.suptitle('Path Length and Curvature Over Time', color='white')
fig_metrics.tight_layout(rect=[0, 0.03, 1, 0.95])
metrics_path = os.path.join(os.path.dirname(__file__), "pathfinder_metrics_history.png")
fig_metrics.savefig(metrics_path, dpi=200, bbox_inches='tight', facecolor=fig_metrics.get_facecolor())
print(f"Saved plot to {metrics_path}")

plt.show()