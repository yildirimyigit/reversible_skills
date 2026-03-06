import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle, Circle
import numpy as np

fig, ax = plt.subplots(figsize=(16, 6), dpi=200)
ax.set_xlim(0, 16)
ax.set_ylim(0, 6)
ax.axis("off")

panel_colors = ["#D8EAF7", "#F4DFC7", "#DDEDCF", "#E5DDF3", "#F6D6D8"]
edge = "#333333"
box_fill = "#F8F8F8"

panels = [
    (0.2, 0.35, 2.8, 5.25, panel_colors[0], "Forward Data Collection"),
    (3.15, 0.35, 2.7, 5.25, panel_colors[1], "State Abstraction"),
    (6.0, 0.35, 3.1, 5.25, panel_colors[2], "Rollback Triage"),
    (9.3, 0.35, 3.5, 5.25, panel_colors[3], "Reverse Policy Learning"),
    (13.0, 0.35, 2.8, 5.25, panel_colors[4], "Evaluation and Deployment"),
]

for x, y, w, h, c, title in panels:
    rr = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5,
        edgecolor=edge,
        facecolor=c,
    )
    ax.add_patch(rr)
    ax.text(
        x + w / 2,
        y + h - 0.28,
        title,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=edge,
    )

def rbox(x, y, w, h, text, fontsize=10, weight="bold"):
    b = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.3,
        edgecolor=edge,
        facecolor=box_fill,
    )
    ax.add_patch(b)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=edge,
    )
    return b

def arrow(x1, y1, x2, y2, text=None, fs=9, rad=0.0, style="-|>"):
    a = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=12,
        connectionstyle=f"arc3,rad={rad}",
        linewidth=1.4,
        color=edge,
    )
    ax.add_patch(a)
    if text:
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + 0.12,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            color=edge,
        )
    return a

# Panel 1
plot_x0, plot_y0, plot_w, plot_h = 0.45, 3.15, 1.3, 1.8
ax.add_patch(Rectangle((plot_x0, plot_y0), plot_w, plot_h, linewidth=1.0, edgecolor="#888", facecolor="white"))
t = np.linspace(0, 1, 100)
y = 0.55 + 0.25 * np.sin(2 * np.pi * (t + 0.12)) + 0.15 * t
ax.plot(plot_x0 + 0.08 + t * (plot_w - 0.16), plot_y0 + 0.1 + y * (plot_h - 0.2), lw=2)
for ti in [0.18, 0.48, 0.82]:
    yi = 0.55 + 0.25 * np.sin(2 * np.pi * (ti + 0.12)) + 0.15 * ti
    ax.add_patch(Circle((plot_x0 + 0.08 + ti * (plot_w - 0.16), plot_y0 + 0.1 + yi * (plot_h - 0.2)), 0.03, color="#555"))
ax.text(plot_x0 + plot_w / 2, plot_y0 + plot_h + 0.08, "Forward trajectory", ha="center", va="bottom", fontsize=9, color=edge)

rbox(1.95, 3.75, 0.75, 0.78, "Forward\nskill", fontsize=10)
rbox(1.95, 2.55, 0.75, 0.78, "Recorder", fontsize=10)
for i in range(3):
    ax.add_patch(Rectangle((0.65 + 0.06 * i, 1.18 + 0.04 * i), 0.62, 0.75, linewidth=1.0, edgecolor=edge, facecolor="white"))
ax.text(0.98, 1.55, "demo\nnpz", ha="center", va="center", fontsize=10, fontweight="bold", color=edge)
ax.text(1.85, 1.55, "actions\nsnapshots\nkeyframes", ha="left", va="center", fontsize=9, color=edge)
arrow(1.75, 4.05, 1.95, 4.14)
arrow(2.33, 3.75, 2.33, 3.33)
arrow(2.15, 2.55, 1.35, 1.92)

# Panel 2
rbox(3.55, 4.0, 1.9, 0.72, "Snapshot utils", fontsize=11)
rbox(3.55, 2.95, 1.9, 0.72, "Compact state  z(s)", fontsize=11)
rbox(3.55, 1.9, 1.9, 0.72, "Keyframes and boundary\ncandidates", fontsize=10)
arrow(4.5, 4.0, 4.5, 3.67)
arrow(4.5, 2.95, 4.5, 2.62)
for i, txt in enumerate(["ee pose", "gripper", "object", "top-K"]):
    ax.add_patch(Rectangle((5.08, 2.98 - 0.17 * i), 0.52, 0.12, linewidth=0.8, edgecolor=edge, facecolor="white"))
    ax.text(5.34, 3.04 - 0.17 * i, txt, ha="center", va="center", fontsize=6.7, color=edge)
arrow(2.98, 1.78, 3.15, 1.78, text="preprocess", fs=8)

# Panel 3
plot2_x0, plot2_y0, plot2_w, plot2_h = 6.25, 3.1, 1.25, 1.85
ax.add_patch(Rectangle((plot2_x0, plot2_y0), plot2_w, plot2_h, linewidth=1.0, edgecolor="#888", facecolor="white"))
t = np.linspace(0, 1, 100)
y = 0.6 + 0.18 * np.sin(2 * np.pi * (t + 0.05)) - 0.22 * t + 0.25 * t**2
ax.plot(plot2_x0 + 0.08 + t * (plot2_w - 0.16), plot2_y0 + 0.1 + y * (plot2_h - 0.2), lw=2)
split_t = 0.62
sx = plot2_x0 + 0.08 + split_t * (plot2_w - 0.16)
ax.plot([sx, sx], [plot2_y0 + 0.08, plot2_y0 + plot2_h - 0.08], lw=1.8, ls="--")
ax.text(plot2_x0 + plot2_w / 2, plot2_y0 + plot2_h + 0.08, "Rollback scores", ha="center", va="bottom", fontsize=9, color=edge)
rbox(7.75, 4.02, 1.0, 0.72, "Rollback\nattempts", fontsize=10)
rbox(7.75, 2.9, 1.0, 0.72, "Consensus\nsplit", fontsize=10)
rbox(6.45, 1.5, 2.2, 0.72, "Boundary snapshot goal", fontsize=11)
diamond = Polygon([[8.3, 2.15], [8.85, 1.8], [8.3, 1.45], [7.75, 1.8]], closed=True, linewidth=1.3, edgecolor=edge, facecolor="white")
ax.add_patch(diamond)
ax.text(8.3, 1.8, "Likely\nreversible?", ha="center", va="center", fontsize=8.5, fontweight="bold", color=edge)
arrow(7.5, 4.02, 7.5, 3.62)
arrow(7.75, 3.25, 7.5, 3.25)
arrow(8.25, 2.9, 8.25, 2.23)
arrow(5.85, 2.2, 6.0, 2.2, text="triage", fs=8)
ax.text(7.05, 0.73, "choose the reversible prefix and\nidentify the irreversible suffix", ha="center", va="center", fontsize=9, color=edge)

# Panel 4
rbox(9.62, 4.12, 1.15, 0.74, "Direct\nreversed\nrollout", fontsize=10)
rbox(10.98, 4.12, 1.15, 0.74, "Reversed\nBC prior", fontsize=10)
rbox(10.98, 2.8, 1.15, 0.74, "Residual\nSAC", fontsize=10)
rbox(12.28, 3.36, 0.32, 0.32, "+", fontsize=16)
rbox(11.95, 1.52, 0.62, 0.72, "Reverse\nexecutor", fontsize=10)
arrow(8.85, 1.92, 9.62, 4.12, text="yes", fs=8, rad=0.1)
arrow(8.85, 1.68, 10.98, 4.12, text="no", fs=8, rad=-0.1)
arrow(11.55, 4.12, 11.55, 3.54)
arrow(11.55, 2.8, 11.55, 2.24)
arrow(12.12, 3.36, 11.55, 3.17)
arrow(12.6, 3.52, 13.0, 3.52, text="policy", fs=8)
arrow(8.3, 1.45, 12.25, 1.88, text="goal = boundary snapshot", fs=8, rad=-0.05)
ax.text(11.1, 0.77, "train only when direct reversal is\nunlikely to succeed", ha="center", va="center", fontsize=9, color=edge)

# Panel 5
rbox(13.35, 4.15, 2.05, 0.72, "Full reverse execution", fontsize=11)
rbox(13.35, 3.0, 2.05, 0.72, "Benchmarks and figures", fontsize=11)
rbox(13.35, 1.85, 2.05, 0.72, "Real robot validation", fontsize=11)
arrow(14.38, 4.15, 14.38, 3.72)
arrow(14.38, 3.0, 14.38, 2.57)
ax.add_patch(Circle((13.68, 0.98), 0.12, edgecolor=edge, facecolor="white", linewidth=1.0))
ax.plot([13.68, 13.68], [0.86, 0.55], color=edge, lw=1.2)
ax.plot([13.68, 13.53], [0.78, 0.66], color=edge, lw=1.2)
ax.plot([13.68, 13.83], [0.78, 0.66], color=edge, lw=1.2)
ax.plot([13.68, 13.57], [0.55, 0.35], color=edge, lw=1.2)
ax.plot([13.68, 13.79], [0.55, 0.35], color=edge, lw=1.2)
ax.text(14.68, 0.98, "ROC / calibration\nsuccess vs steps\ngeneralization", ha="center", va="center", fontsize=8.5, color=edge)

for x1, x2, y in [(3.0, 3.15, 3.0), (5.85, 6.0, 3.0), (9.1, 9.3, 3.0), (12.8, 13.0, 3.0)]:
    arrow(x1, y, x2, y)

ax.text(8, 5.92, "PLAN-A: From Forward Demonstrations to Reverse Skill Execution", ha="center", va="top", fontsize=16, fontweight="bold", color=edge)

png_path = "workflow.png"
svg_path = "workflow.svg"
plt.savefig(png_path, bbox_inches="tight", facecolor="white")
plt.savefig(svg_path, bbox_inches="tight", facecolor="white")
print(png_path)
print(svg_path)