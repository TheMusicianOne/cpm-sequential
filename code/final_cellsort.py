import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import lattice
import time

# --------------------------
# Parameters
# --------------------------
N = 40      # lattice size
T = 15           # temperature
MCS_TOTAL = 5000  # run time

# Adhesion: self low (stick), hetero high (repel)
adhesion_list = [
    [5, 20, 0],   # RED vs {red,blue,bg}
    [20, 5, 0],   # BLUE vs {red,blue,bg}
    [0,  0, 0]    # BACKGROUND
]

# Target constraints: small cells
target_volume = 20
volume_strength = 10
target_perimeter = 5
perimeter_strength = 2

targets = {
    0: [(target_volume, volume_strength), (target_perimeter, perimeter_strength)],  # RED
    1: [(target_volume, volume_strength), (target_perimeter, perimeter_strength)],  # BLUE
    2: [(float('inf'), 0), (0, 0)]  # BACKGROUND
}

# --------------------------
# Initialize lattice
# --------------------------
lat = lattice.Lattice(N=N, T=T, adhesion_list=adhesion_list, targets=targets)

np.random.seed(0)
cell_id = 10

# place ~80 random single-pixel cells (half red, half blue)
n_cells = N*N
cell_id = 4  # start above background=2
for x in range(N):
    for y in range(N):
        color = np.random.randint(0, 2)   # 0=red, 1=blue
        lat.add_cell(cid=cell_id, ctype=color, positions=[(x, y)])
        cell_id += 1


print(f"Seeded {n_cells} single-pixel cells on {N}x{N} grid")

# --------------------------
# Metrics
# --------------------------
def calculate_mixing_index():
    diff_pairs, total_pairs = 0, 0
    for i in range(lat.N):
        for j in range(lat.N):
            cid = lat.lattice[i, j]
            if cid not in lat.cells: 
                continue
            ctype = lat.cells[cid].ctype
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = (i+di) % lat.N, (j+dj) % lat.N
                nid = lat.lattice[ni, nj]
                if nid in lat.cells:
                    ntype = lat.cells[nid].ctype
                    total_pairs += 1
                    if ctype != ntype: diff_pairs += 1
    return diff_pairs / max(1, total_pairs)

# --------------------------
# Visualization
# --------------------------
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))
colors = ['red','blue','lightgray']
cmap = mcolors.ListedColormap(colors)

mixing_hist = []
time_hist = []
red_hist = []
blue_hist = []
mcs = 0

def update_plot():
    ax1.clear()
    visual = np.full((N,N), 2)  # background
    for i in range(N):
        for j in range(N):
            cid = lat.lattice[i,j]
            if cid in lat.cells:
                visual[i,j] = lat.cells[cid].ctype
    ax1.imshow(visual, cmap=cmap, vmin=0, vmax=2)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title(f"MCS={mcs}")

    ax2.clear()
    ax2.plot(time_hist, mixing_hist, 'k-', label="Mixing index")
    ax2.plot(time_hist, red_hist, 'r-', label="Red volume")
    ax2.plot(time_hist, blue_hist, 'b-', label="Blue volume")
    ax2.set_xlabel("MCS")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.pause(0.01)

# --------------------------
# Metrics
# --------------------------
def get_volumes():
    red = sum(c.volume for c in lat.cells.values() if c.ctype == 0)
    blue = sum(c.volume for c in lat.cells.values() if c.ctype == 1)
    return red, blue


paused = True  # start paused until space is pressed

def on_key(event):
    global paused
    if event.key == " ":
        paused = not paused  # toggle pause/resume

fig.canvas.mpl_connect("key_press_event", on_key)

# Show initial state
update_plot()
plt.pause(0.01)

# --------------------------
# Simulation loop
# --------------------------
for step in range(MCS_TOTAL):
    if paused:
        plt.pause(0.1)
        continue

    lat.monte_carlo_step()
    mcs += 1
    mix = calculate_mixing_index()
    red, blue = get_volumes()
    mixing_hist.append(mix)
    red_hist.append(red)
    blue_hist.append(blue)
    time_hist.append(mcs)

    if mcs % 10 == 0:   # update plot less often to speed up
        update_plot()

    if mcs % 100 == 0:
        print(f"Step {mcs}, mixing={mix:.3f}, red={red}, blue={blue}")



plt.show()
