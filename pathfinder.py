import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from scipy.interpolate import BSpline
import random

X_history = []

# ── Config ─────────────────────────────────────────────────────────────────
START = (5.0,  5.0,  math.pi * 0.25)   # slight upward heading
GOAL  = (95.0, 95.0, math.pi * 0.25)
RHO   = 10.0                             # turning radius (hardcoded, kept small)

# Two horizontal walls — clear corridor exists ABOVE (y > 64) and BELOW (y < 36)
WALLS = [(15, 30, 30, 72), (55, 42, 85, 60)]

def build_bspline(control_pts, degree=3, n_samples=200):
    control_pts = np.array(control_pts)

    n = len(control_pts)
    k = degree

    # clamped knot vector
    knots = np.concatenate((
        np.zeros(k),
        np.linspace(0, 1, n - k + 1),
        np.ones(k)
    ))

    t = np.linspace(0, 1, n_samples)

    spline_x = BSpline(knots, control_pts[:,0], k)
    spline_y = BSpline(knots, control_pts[:,1], k)

    x = spline_x(t)
    y = spline_y(t)

    dx = spline_x.derivative(1)(t)
    dy = spline_y.derivative(1)(t)

    ddx = spline_x.derivative(2)(t)
    ddy = spline_y.derivative(2)(t)

    return x, y, dx, dy, ddx, ddy

def compute_curvature(dx, dy, ddx, ddy):
    num = np.abs(dx * ddy - dy * ddx)
    denom = (dx**2 + dy**2)**1.5 + 1e-6
    return num / denom

def sample_seg(x, y, th, L, kind, rho, n=80):
    pts = []
    if kind == 'S':
        ss = np.linspace(0, L, max(3, int(L/0.3)+1))
        for s in ss: pts.append((x+s*math.cos(th), y+s*math.sin(th)))
        x += L*math.cos(th); y += L*math.sin(th)
    elif kind == 'L':
        cx = x - rho*math.sin(th); cy = y + rho*math.cos(th)
        da = L / rho
        ns = max(3, int(da*rho/0.3)+1)
        for a in np.linspace(0, da, ns):
            phi = th + a; pts.append((cx+rho*math.sin(phi), cy-rho*math.cos(phi)))
        th += da; x, y = pts[-1]
    else:  # R
        cx = x + rho*math.sin(th); cy = y - rho*math.cos(th)
        da = L / rho
        ns = max(3, int(da*rho/0.3)+1)
        for a in np.linspace(0, da, ns):
            phi = th - a; pts.append((cx-rho*math.sin(phi), cy+rho*math.cos(phi)))
        th -= da; x, y = pts[-1]
    return pts, x, y, th

# ── Collision helpers ───────────────────────────────────────────────────────
def pt_in_wall(px, py, wall, mg=0.3):
    return wall[0]-mg<=px<=wall[2]+mg and wall[1]-mg<=py<=wall[3]+mg

def seg_clear(ax, ay, bx, by, walls, mg=0.5):
    """Check a line segment is clear of all walls."""
    n = max(20, int(math.hypot(bx-ax, by-ay)/0.5))
    for t in np.linspace(0, 1, n):
        px, py = ax+t*(bx-ax), ay+t*(by-ay)
        for w in walls:
            if pt_in_wall(px, py, w, mg): return False
    return True

def arc_clear(pts, walls):
    arr = np.array(pts)
    for px, py in arr:
        for w in walls:
            if pt_in_wall(px, py, w): return False
    return True

# ── RRT to find collision-free skeleton ────────────────────────────────────
def rrt(start_xy, goal_xy, walls, n_iter=8000, step=6.0, goal_r=8.0, mg=RHO+0.5):
    rng = np.random.default_rng(7)
    nodes = [np.array(start_xy, float)]
    parent = [-1]
    goal_arr = np.array(goal_xy, float)
    for _ in range(n_iter):
        s = goal_arr if rng.random()<0.12 else rng.uniform(0,100,2)
        dists = [np.linalg.norm(n-s) for n in nodes]
        ni = int(np.argmin(dists))
        nn = nodes[ni]
        d = np.linalg.norm(s-nn)
        if d<1e-6: continue
        newp = nn + (s-nn)/d * min(step, d)
        if not seg_clear(nn[0],nn[1],newp[0],newp[1], walls, mg): continue
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
rrt_path = rrt((START[0],START[1]), (GOAL[0],GOAL[1]), WALLS)
if rrt_path is None:
    raise RuntimeError("RRT failed to find a path — try increasing n_iter")
print(f"  RRT: {len(rrt_path)} nodes")

# ── Prune collinear points ──────────────────────────────────────────────────
def prune(path, walls, mg=RHO+0.5):
    """Remove waypoints that can be skipped without collision."""
    pruned = [path[0]]
    i = 0
    while i < len(path)-1:
        j = len(path)-1
        while j > i+1:
            if seg_clear(path[i][0],path[i][1], path[j][0],path[j][1], walls, mg):
                break
            j -= 1
        pruned.append(path[j]); i = j
    return pruned

rrt_pruned = prune(rrt_path, WALLS)
print(f"  After pruning: {len(rrt_pruned)} nodes")

# ── Assign headings ─────────────────────────────────────────────────────────
def assign_headings(pts, start_th, goal_th):
    ths = []
    for i in range(len(pts)-1):
        dx=pts[i+1][0]-pts[i][0]; dy=pts[i+1][1]-pts[i][1]
        ths.append(math.atan2(dy,dx))
    ths.append(goal_th)
    ths[0] = start_th
    # Smooth interior headings: average of incoming and outgoing
    for i in range(1, len(pts)-1):
        dx_in  = pts[i][0]-pts[i-1][0]; dy_in  = pts[i][1]-pts[i-1][1]
        dx_out = pts[i+1][0]-pts[i][0]; dy_out = pts[i+1][1]-pts[i][1]
        th_in  = math.atan2(dy_in,  dx_in)
        th_out = math.atan2(dy_out, dx_out)
        ths[i] = math.atan2(math.sin(th_in)+math.sin(th_out),
                             math.cos(th_in)+math.cos(th_out))
    return ths

def particle_swarm(f, x_lb, x_ub, n=10, inertia=0.5, self_influence=1.8, social_influence=1.8, max_vel=1.0, f_tol=1e-6):
    global nfev
    global X_history
    nfev = 0
    X_history = []

    k = 0
    X = np.zeros((n, len(x_lb)))
    dX = np.zeros((n, len(x_lb)))

    for i in range(n):
        X[i,:] = x_lb + (x_ub - x_lb) * np.random.rand(len(x_lb))
        dX[i,:] = (np.random.rand(len(x_lb))*2 - 1) * max_vel     
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
    return r_x, f(r_x), nfev

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

def build_spline_from_x(x):
    pts = [START[:2]]

    for i in range(0, len(x), 2):
        pts.append((x[i], x[i+1]))

    pts.append(GOAL[:2])

    x_s, y_s, dx, dy, ddx, ddy = build_bspline(pts)

    curvature = compute_curvature(dx, dy, ddx, ddy)

    # arc length
    ds = np.sqrt(dx**2 + dy**2)
    length = np.sum(ds)

    # collision
    collision = False
    for px, py in zip(x_s, y_s):
        for w in WALLS:
            if pt_in_wall(px, py, w):
                collision = True

    return x_s, y_s, length, curvature, collision

def objective(x):
    x_s, y_s, length, curvature, collision = build_spline_from_x(x)

    # curvature constraint
    kappa_max = 1.0 / RHO
    curvature_violation = np.maximum(0, curvature - kappa_max)

    curvature_penalty = np.sum(curvature_violation**2) * 1e4

    collision_penalty = 1e6 if collision else 0

    return length + curvature_penalty + collision_penalty

# ── Build bounds from RRT path ─────────────────────────────────────────────
margin = 10.0
lb = []
ub = []

for (x, y) in rrt_pruned[1:-1]:
    lb.extend([max(0, x - margin), max(0, y - margin)])
    ub.extend([min(100, x + margin), min(100, y + margin)])

lb = np.array(lb)
ub = np.array(ub)

print("Optimization dim:", len(lb))

# ── Run PSO ────────────────────────────────────────────────────────────────
x_opt, f_opt, _ = particle_swarm(
    objective,
    lb,
    ub,
    n=20,
    max_vel=5.0   # IMPORTANT: reduced from 100
)

visualize_swarm(X_history, lb, ub, objective)

print("Optimized cost:", f_opt)

x_s, y_s, length, curvature, collision = build_spline_from_x(x_opt)

print("Length:", length)
print("Max curvature:", np.max(curvature))
print("Allowed:", 1.0 / RHO)

# ── Plot optimized result ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9,9))
ax.set_facecolor('#0d0d1a')

# obstacles
for w in WALLS:
    ax.add_patch(plt.Polygon(
        [(w[0],w[1]),(w[2],w[1]),(w[2],w[3]),(w[0],w[3])],
        closed=True, fc='#922b21', ec='#e74c3c', alpha=0.9
    ))

# spline path
points = np.array([x_s, y_s]).T.reshape(-1,1,2)
segs = np.concatenate([points[:-1], points[1:]], axis=1)

lc = LineCollection(segs, cmap='plasma', lw=2.5)
lc.set_array(curvature)
ax.add_collection(lc)

# control points
wps = [(x_opt[i], x_opt[i+1]) for i in range(0, len(x_opt), 2)]
if wps:
    wx, wy = zip(*wps)
    ax.scatter(wx, wy, c='yellow', s=70, edgecolors='white')

# start/goal
ax.plot(*START[:2],'o',color='#2ecc71',ms=12)
ax.plot(*GOAL[:2],'*',color='#e74c3c',ms=16)

ax.set_xlim(0,100)
ax.set_ylim(0,100)
ax.set_aspect('equal')

plt.title(f"B-Spline Path | Length={length:.1f}")
plt.show()