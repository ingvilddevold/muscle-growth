"""
Demo Script: Signaling Model with Single Exercise Protocol
"""

from pathlib import Path
from musclex.exercise_model import *
from musclex.protocol import *
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

# 1. Setup
config = Path(__file__).parents[1] / "config_files/exercise_eq_reduced_k1.yml"
results = Path(__file__).parents[1] / "results/demos/signaling_demo"
results.mkdir(parents=True, exist_ok=True)

# 2. Define the exercise protocol
# Number of bouts: 7
# Duration of bout: 1 hour
# Time between bouts: 47 hours
# End time: 15 days
# Intensity: 0.8 (80% of max)
protocol = RegularExercise(
    N=7, exercise_duration=1, growth_duration=47, end_time=15 * 24, intensity=0.8
)

# Print protocol events
fmt = lambda t: f"Day {int(t//24)}, {t%24:04.1f}h"
print("\nExercise Protocol:")
print(f"{'TYPE':<15} {'START':<15} {'END':<15} {'DURATION'}")
for e in protocol.events:
    print(
        f"{type(e).__name__:<15} {fmt(e.start_time):<15} {fmt(e.end_time):<15} {e.end_time-e.start_time:.1f}h"
    )

# 3. Initialize and Run the Model
model = ExerciseModel(protocol, str(config), results)
print(f"\nRunning simulation for {protocol.events[-1].end_time/24:.1f} days...")
model.simulate()
print("Simulation complete.")

# 4. Visualize Results (5-Panel Grid)
state_names = ["IGF1", "AKT", "FOXO", "mTOR", "Myofibrils (N)"]
t_days = model.t / 24

fig = plt.figure(figsize=(7, 7))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.2])

axes = [
    plt.subplot(gs[0, 0]),  # IGF1
    plt.subplot(gs[0, 1]),  # AKT
    plt.subplot(gs[1, 0]),  # FOXO
    plt.subplot(gs[1, 1]),  # mTOR
    plt.subplot(gs[2, :]),  # Myofibrils
]

for i, ax in enumerate(axes):
    # Plot the state data
    ax.plot(t_days, model.y[i, :], color="#1f77b4", linewidth=2)

    # Formatting
    ax.set_title(state_names[i], fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    # ax.set_xlim(0, 15)

    ax.set_xlabel("Time (days)")

plt.suptitle("Signaling Dynamics", fontsize=14, y=0.96)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])

output_path = results / "demo_signaling_single_protocol.png"
plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
print(f"Results plotted to {output_path}")

plt.show()
