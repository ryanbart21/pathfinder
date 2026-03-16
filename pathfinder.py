import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ── Config ─────────────────────────────────────────────────────────────────
START = (5.0,  5.0,  math.pi * 0.25)   # slight upward heading
GOAL  = (95.0, 95.0, math.pi * 0.25)
RHO   = 4.0                             # turning radius (hardcoded, kept small)

# Two horizontal walls — clear corridor exists ABOVE (y > 64) and BELOW (y < 36)
WALLS = [(15, 30, 30, 72), (55, 42, 85, 60)]

# ── Dubins solver ───────────────────────────────────────────────────────────
def mod2pi(a): return a % (2 * math.pi)

def _dubins_candidates(alpha, beta, d):
    sa, ca = math.sin(alpha), math.cos(alpha)
    sb, cb = math.sin(beta),  math.cos(beta)
    out = []
    v = 2+d*d-2*(ca*cb+sa*sb-d*(sa-sb))
    if v>=0: p=math.sqrt(v); th=math.atan2(cb-ca,d+sa-sb); out.append((mod2pi(-alpha+th),p,mod2pi(beta-th),'LSL'))
    v = 2+d*d-2*(ca*cb+sa*sb-d*(sb-sa))
    if v>=0: p=math.sqrt(v); th=math.atan2(ca-cb,d-sa+sb); out.append((mod2pi(alpha-th),p,mod2pi(-beta+th),'RSR'))
    v = -2+d*d+2*(ca*cb+sa*sb+d*(sa+sb))
    if v>=0: p=math.sqrt(v); th=math.atan2(-ca-cb,d+sa+sb)-math.atan2(-2,p); out.append((mod2pi(-alpha+th),p,mod2pi(-beta+th),'LSR'))
    v = d*d-2+2*(ca*cb+sa*sb-d*(sa+sb))
    if v>=0: p=math.sqrt(v); th=math.atan2(ca+cb,d-sa-sb)-math.atan2(2,p); out.append((mod2pi(alpha-th),p,mod2pi(beta-th),'RSL'))
    v = (6-d*d+2*(ca*cb+sa*sb+d*(sa-sb)))/8
    if abs(v)<=1:
        p=mod2pi(2*math.pi-math.acos(np.clip(v,-1,1))); th=math.atan2(ca-cb,d-sa+sb)
        t=mod2pi(alpha-th+p/2); out.append((t,p,mod2pi(alpha-beta-t+p),'RLR'))
    v = (6-d*d+2*(ca*cb+sa*sb-d*(sa-sb)))/8
    if abs(v)<=1:
        p=mod2pi(2*math.pi-math.acos(np.clip(v,-1,1))); th=math.atan2(-ca+cb,d+sa-sb)
        t=mod2pi(-alpha+th+p/2); out.append((t,p,mod2pi(beta-alpha-t+p),'LRL'))
    return out

def dubins(q0, q1, rho):
    dx=(q1[0]-q0[0])/rho; dy=(q1[1]-q0[1])/rho; d=math.sqrt(dx*dx+dy*dy)
    if d<1e-6: return None
    th=mod2pi(math.atan2(dy,dx))
    cands=_dubins_candidates(mod2pi(q0[2]-th), mod2pi(q1[2]-th), d)
    if not cands: return None
    t,p,q,w=min(cands, key=lambda c:c[0]+c[1]+c[2])
    return (t+p+q)*rho, t*rho, p*rho, q*rho, w

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

def sample_dubins(q0, t, p, q, word, rho):
    all_pts=[]; x,y,th=q0
    for L,kind in zip([t,p,q], word):
        pts,x,y,th = sample_seg(x,y,th,L,kind,rho)
        all_pts.extend(pts)
    return np.array(all_pts)

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

ths = assign_headings(rrt_pruned, START[2], GOAL[2])
configs = [(rrt_pruned[i][0], rrt_pruned[i][1], ths[i]) for i in range(len(rrt_pruned))]
# Force exact start and goal configs
configs[0]  = START
configs[-1] = GOAL

# ── Build Dubins path and verify ────────────────────────────────────────────
all_pts = []
total_len = 0
collision = False
print("\nDubins segments:")
for i in range(len(configs)-1):
    res = dubins(configs[i], configs[i+1], RHO)
    if res is None:
        print(f"  Seg {i}: no path found"); continue
    length, t, p, q, word = res
    total_len += length
    pts = sample_dubins(configs[i], t, p, q, word, RHO)
    if not arc_clear(pts, WALLS): collision = True
    all_pts.append(pts)
    print(f"  Seg {i}: {word}  len={length:.1f}")

# Concatenate segments, dropping the duplicate first point of each segment
# (which equals the last point of the previous one) to avoid LineCollection
# drawing a stray line across any floating-point gap at joins.
continuous = [all_pts[0]]
for seg in all_pts[1:]:
    continuous.append(seg[1:])   # skip index 0 — it duplicates previous end
# Force exact goal as final point
continuous.append(np.array([[GOAL[0], GOAL[1]]]))
smooth = np.vstack(continuous)
status = "⚠ Arc clips obstacle" if collision else "✓ Collision-free"
print(f"\n{status}  |  Total length: {total_len:.1f}  |  rho={RHO}")

# ── Plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9,9))
ax.set_facecolor('#0d0d1a'); fig.patch.set_facecolor('#0d0d1a')

# Obstacles
for w in WALLS:
    ax.add_patch(plt.Polygon([(w[0],w[1]),(w[2],w[1]),(w[2],w[3]),(w[0],w[3])],
        closed=True, fc='#922b21', ec='#e74c3c', lw=1.5, alpha=0.9, zorder=2))
    ax.text((w[0]+w[2])/2,(w[1]+w[3])/2,'WALL',ha='center',va='center',
            color='#faa',fontsize=8,fontweight='bold',zorder=3)

# Smooth Dubins path
pts_r = smooth.reshape(-1,1,2)
segs  = np.concatenate([pts_r[:-1], pts_r[1:]], axis=1)
lc = LineCollection(segs, cmap='plasma', lw=2.5, zorder=4, alpha=0.95)
lc.set_array(np.linspace(0,1,len(segs)))
ax.add_collection(lc)

# Waypoints (pruned RRT nodes, excluding start/goal)
wx = [c[0] for c in configs[1:-1]]
wy = [c[1] for c in configs[1:-1]]
ax.scatter(wx, wy, c='#f39c12', s=65, zorder=6,
           marker='D', edgecolors='white', lw=0.8, label='Waypoints')

# Start / Goal
ax.plot(*START[:2],'o',color='#2ecc71',ms=13,zorder=7,mec='white',mew=1.5,label='Start')
ax.plot(*GOAL[:2], '*',color='#e74c3c',ms=17,zorder=7,mec='white',mew=1.5,label='Goal')
ax.text(START[0]+1.5, START[1]-4,'START',color='#2ecc71',fontsize=8,fontweight='bold',zorder=8)
ax.text(GOAL[0]-10,   GOAL[1]+2, 'GOAL', color='#e74c3c',fontsize=8,fontweight='bold',zorder=8)

ax.set_xlim(0,100); ax.set_ylim(0,100); ax.set_aspect('equal')
ax.grid(True,color='#1a1a2e',lw=0.6)
ax.tick_params(colors='#555')
for sp in ax.spines.values(): sp.set_edgecolor('#333')
ax.legend(facecolor='#16213e',edgecolor='#444',labelcolor='white',fontsize=9,loc='lower right')
ax.set_title("Dubins Path  —  RRT skeleton + arc smoothing",color='white',fontsize=13,pad=10)
col_color = '#aaa' if not collision else '#e74c3c'
ax.text(1,98,f"{status}   |   Length: {total_len:.1f}   |   ρ = {RHO}",
        color=col_color,fontsize=8,va='top')

plt.tight_layout()
plt.savefig('dubins_path.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
print("Saved dubins_path.png")