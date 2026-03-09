import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle, Circle
import numpy as np
from pathlib import Path

fig, ax = plt.subplots(figsize=(16.8, 6.9), dpi=220)
ax.set_xlim(0, 16.8)
ax.set_ylim(0, 6.9)
ax.axis("off")

edge = "#3A3A3A"
box_fill = "#F7F7F7"
panel_colors = ["#DCE9F3", "#F0DFC9", "#DCE7D4", "#E3DCEF"]

panels = [
    (0.30, 0.42, 3.45, 5.95, panel_colors[0], "Forward Data\nCollection"),
    (3.95, 0.42, 3.30, 5.95, panel_colors[1], "State Abstraction"),
    (7.45, 0.42, 3.95, 5.95, panel_colors[2], "Rollback Triage"),
    (11.60, 0.42, 4.60, 5.95, panel_colors[3], "Reverse Policy Learning"),
]

for x, y, w, h, c, title in panels:
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5, edgecolor=edge, facecolor=c
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2, y + h - 0.38, title,
        ha="center", va="top",
        fontsize=15, fontweight="bold", color=edge, linespacing=1.0
    )

def rbox(x, y, w, h, text, fontsize=10, weight="bold", fc=box_fill):
    b = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.25, edgecolor=edge, facecolor=fc
    )
    ax.add_patch(b)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize, fontweight=weight, color=edge, linespacing=1.0
    )
    return b

def arrow(x1, y1, x2, y2, text=None, fs=9, rad=0.0, dx=0.0, dy=0.0, lw=1.4):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=12,
        connectionstyle=f"arc3,rad={rad}",
        linewidth=lw, color=edge
    )
    ax.add_patch(a)
    if text:
        ax.text(
            (x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy,
            text, ha="center", va="center", fontsize=fs, color=edge
        )

# ---------------- Panel 1 ----------------
px, py, pw, ph = 0.62, 3.05, 1.62, 2.10
ax.add_patch(Rectangle((px, py), pw, ph, linewidth=1.0, edgecolor="#8A8A8A", facecolor="white"))

t = np.linspace(0, 1, 120)
curve = 0.60 + 0.22 * np.sin(2 * np.pi * (t + 0.15)) + 0.10 * t
ax.plot(px + 0.10 + t * (pw - 0.20), py + 0.12 + curve * (ph - 0.24), lw=2.3)

for ti in [0.16, 0.50, 0.82]:
    yi = 0.60 + 0.22 * np.sin(2 * np.pi * (ti + 0.15)) + 0.10 * ti
    ax.add_patch(Circle(
        (px + 0.10 + ti * (pw - 0.20), py + 0.12 + yi * (ph - 0.24)),
        0.03, color="#575757"
    ))

ax.text(px + pw / 2, py + ph + 0.10, "Forward trajectory",
        ha="center", va="bottom", fontsize=10, color=edge)

rbox(2.42, 3.78, 0.98, 0.98, "Forward\nskill", fontsize=10)
rbox(2.42, 2.52, 0.98, 0.98, "Recorder", fontsize=10)

for i in range(3):
    ax.add_patch(Rectangle(
        (0.90 + 0.06 * i, 1.18 + 0.05 * i),
        0.74, 0.86,
        linewidth=1.0, edgecolor=edge, facecolor="white"
    ))

ax.text(1.27, 1.61, "demo\nnpz", ha="center", va="center",
        fontsize=10, fontweight="bold", color=edge)
ax.text(2.30, 1.53, "actions\nsnapshots\nkeyframes",
        ha="left", va="center", fontsize=10, color=edge)

arrow(2.24, 4.10, 2.42, 4.22)
arrow(2.91, 3.78, 2.91, 3.48)
arrow(2.42, 2.54, 1.64, 1.92)

arrow(3.75, 2.95, 3.95, 2.95)
ax.text(3.85, 3.12, "preprocess", ha="center", va="center",
        fontsize=9, color=edge)

# ---------------- Panel 2 ----------------
rbox(4.35, 4.22, 2.55, 0.84, "Snapshot utils", fontsize=12)
rbox(4.35, 2.97, 2.55, 0.84, "Compact state\nz(s)", fontsize=12)
rbox(4.35, 1.72, 2.55, 0.84, "Keyframes\nand boundary\ncandidates", fontsize=11)

arrow(5.62, 4.22, 5.62, 3.81)
arrow(5.62, 2.97, 5.62, 2.56)

# cleaner than four tiny overlapping boxes
ax.text(
    6.7, 3.0,
    "ee pose\ngripper\ntop-K\nobject",
    ha="center", va="center",
    fontsize=8.3, color=edge,
    bbox=dict(
        boxstyle="round,pad=0.18,rounding_size=0.04",
        facecolor="white", edgecolor=edge, linewidth=0.8
    )
)

arrow(7.25, 2.95, 7.45, 2.95)
ax.text(7.35, 3.12, "triage", ha="center", va="center",
        fontsize=9, color=edge)

# ---------------- Panel 3 ----------------
rx, ry, rw, rh = 7.82, 3.20, 1.65, 2.15
ax.add_patch(Rectangle((rx, ry), rw, rh, linewidth=1.0, edgecolor="#8A8A8A", facecolor="white"))

u = np.linspace(0, 1, 140)
curve2 = 0.62 + 0.18 * np.sin(2 * np.pi * (u + 0.08)) - 0.16 * u + 0.16 * u**2
ax.plot(rx + 0.10 + u * (rw - 0.20), ry + 0.12 + curve2 * (rh - 0.24), lw=2.4)

sx = rx + 0.10 + 0.60 * (rw - 0.20)
ax.plot([sx, sx], [ry + 0.10, ry + rh - 0.10],
        lw=2.2, ls="--", color="#29963A")

ax.text(rx + rw / 2, ry + rh + 0.10, "Rollback scores",
        ha="center", va="bottom", fontsize=10, color=edge)

rbox(9.78, 4.02, 1.18, 0.88, "Rollback\nattempts", fontsize=10)
rbox(9.78, 2.78, 1.18, 0.88, "Consensus\nsplit", fontsize=10)
rbox(7.75, 1.72, 2.10, 0.76, "Boundary\nsnapshot goal", fontsize=11)

diamond = Polygon(
    [[10.58, 1.52], [11.16, 1.10], [10.58, 0.68], [10.0, 1.10]],
    closed=True, linewidth=1.25, edgecolor=edge, facecolor="white"
)
ax.add_patch(diamond)
ax.text(10.58, 1.15, "Likely\nreversible?",
        ha="center", va="center", fontsize=9.0,
        fontweight="bold", color=edge)

arrow(9.47, 4.46, 9.78, 4.46)
arrow(10.37, 4.02, 10.37, 3.66)
arrow(9.78, 3.22, 9.54, 3.22)
arrow(10.37, 2.78, 10.37, 2.48)

# boundary -> decision
arrow(9.85, 1.72, 10.40, 1.42)

ax.text(
    9.44, 0.66,
    "choose the reversible\nprefix and identify the irreversible suffix",
    ha="center", va="center",
    fontsize=10, color=edge, linespacing=1.0
)

arrow(11.40, 2.95, 11.60, 2.95)

# ---------------- Panel 4 ----------------
rbox(12.00, 4.18, 1.50, 0.98, "Direct\nreversed\nrollout", fontsize=10)
rbox(14.00, 4.18, 1.50, 0.98, "Reversed\nBC prior", fontsize=10)
rbox(14.00, 2.80, 1.50, 0.98, "Residual\nSAC", fontsize=10)
rbox(14.10, 1.62, 1.35, 0.86, "Reverse\nexecutor", fontsize=10)

arrow(11.06, 1.24, 12.00, 4.18, rad=0.10)
ax.text(11.82, 2.95, "yes", fontsize=9, color=edge)

arrow(11.06, 1.04, 14.00, 4.18, rad=-0.06)
ax.text(12.43, 2.55, "no", fontsize=9, color=edge)

# direct reversed rollout -> executor
arrow(12.75, 4.18, 14.55, 2.48, rad=-0.10)

# BC -> residual -> executor
arrow(14.75, 4.18, 14.75, 3.78)
arrow(14.75, 2.80, 14.75, 2.48)

# boundary goal -> executor
arrow(9.85, 1.96, 14.10, 2.02, rad=-0.02)
ax.text(12.35, 1.70, "goal = boundary\nsnapshot",
        fontsize=9, color=edge)

ax.text(
    13.95, 0.92,
    "train only when direct reversal is\nunlikely to succeed",
    ha="center", va="center",
    fontsize=10, color=edge, linespacing=1.0
)

# ---------------- Main title ----------------
ax.text(
    8.40, 6.60,
    "From Forward Demonstrations to Reverse Skill Execution",
    ha="center", va="center",
    fontsize=19, fontweight="bold", color=edge
)

# save next to the script if this is run as a script,
# otherwise save into the current working directory
try:
    root_path = Path(__file__).resolve().parent
except NameError:
    root_path = Path.cwd()

png_path = root_path / "workflow.png"
svg_path = root_path / "workflow.svg"
pdf_path = root_path / "workflow.pdf"

plt.savefig(png_path, bbox_inches="tight", facecolor="white")
plt.savefig(svg_path, bbox_inches="tight", facecolor="white")
plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")

print(png_path)
print(svg_path)
print(pdf_path)