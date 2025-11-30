import matplotlib.pyplot as plt

# Setup
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.size"] = 8
plt.rcParams["hatch.linewidth"] = 2  # Make the hatch pattern thicker
plt.rcParams["svg.fonttype"] = "none"  # To make text editable in SVG

# Define days for each protocol (0=Mon, 1=Tue, ...)
days_mwf = [0, 2, 4]  # Mon, Wed, Fri
days_every3 = [0, 3, 6]  # Mon, Thu, Sun
day_weekly = [0]  # Mon

# Setup figure
fig, ax = plt.subplots(figsize=(2.0, 1.2))
plt.subplots_adjust(right=0.6) # make room for legend

# Colors
c_mwf = '#5e81ac'      # Muted Blue
c_every3 = '#a3be8c'   # Muted Green
c_weekly = '#d08770'   # Muted Orange

# --- Plotting ---

# 1. "Every three days"
ax.bar(
    days_every3,
    20, # 20% intensity
    color=c_every3,
    alpha=1.0,
    width=0.6,
    label="Every three\ndays",
    align="center",
)

# 2. "MWF"
ax.bar(days_mwf, 80, color=c_mwf, alpha=1.0, width=0.25, label="MWF", align="center")

# 3. "Weekly"
# Overlaying the MWF bar on Monday
ax.bar(
    day_weekly,
    80,
    bottom=0,
    facecolor="none",
    edgecolor=c_weekly,
    hatch="---",
    linewidth=0,
    width=0.25,
    label="Weekly",
    align="center",
)

# --- Customization ---
ax.set_ylim(0, 105)
ax.set_xlim(-0.6, 6.6)

# Spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_linewidth(1.0)
ax.spines["bottom"].set_color("black")

# X-Axis
ax.set_xticks(range(7))
ax.set_xticklabels(["M", "T", "W", "T", "F", "S", "S"], fontsize=8, color="black")
ax.tick_params(axis="x", width=0.5, length=3, color="black")

# Y-Axis
ax.set_yticks([])
ax.set_ylabel("Intensity", fontsize=8, labelpad=0, color="black")

# Legend - Center Right
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [
    Patch(facecolor=c_mwf, label="MWF"),
    Patch(facecolor=c_every3, label="Every three\ndays"),
    Patch(
        facecolor="none",
        edgecolor=c_weekly,
        hatch="---",
        linewidth=0,
        label="Weekly",
    ),
]
legend_elements = [
    Line2D([0], [0], color=c_mwf, lw=3, label='MWF'),
    Line2D([0], [0], color=c_every3, lw=3, label='Every three\ndays'),
    Line2D([0], [0], color=c_weekly, lw=3, linestyle=':', label='Weekly')
]

# Position legend to the right of the plot
ax.legend(
    handles=legend_elements,
    loc="center left",
    bbox_to_anchor=(0.95, 0.4),
    frameon=False,
    ncol=1,
    fontsize=8,
    handlelength=1.5,
)

# Save
#plt.savefig("exercise_protocols_conceptual.png", dpi=300, bbox_inches="tight")
plt.savefig("exercise_protocols_conceptual.svg", bbox_inches="tight")
plt.show()
