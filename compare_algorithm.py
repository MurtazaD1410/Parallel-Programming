import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the combined dataset
df_combined = pd.read_csv("matrix_performance_combined.csv")

# Ensure processors column is numeric
df_combined["processors"] = pd.to_numeric(df_combined["processors"], errors="coerce")

# Choose a single representative processor count for clarity
chosen_processor = 400

# Define colors and markers for each algorithm
colors = {"Cannon": "#1f77b4", "Row": "#ff7f0e", "Fox": "#2ca02c"}
markers = {"Cannon": "o", "Row": "s", "Fox": "^"}

# Create figure
plt.figure(figsize=(10, 6))

# Plot only for the selected processor count
for algorithm in ["Cannon", "Row", "Fox"]:
    data = df_combined[(df_combined["algorithm"] == algorithm) & 
                       (df_combined["processors"] == chosen_processor)]
    if not data.empty:
        plt.plot(data["dimension"], data["time"],
                 marker=markers[algorithm],
                 color=colors[algorithm],
                 linestyle="-",
                 linewidth=2,
                 label=f"{algorithm} ({chosen_processor} procs)",
                 markersize=6)

# Configure plot
plt.title("Simplified Matrix Multiplication Algorithm Comparison", pad=15)
plt.xlabel("Matrix Dimension", fontsize=12)
plt.ylabel("Time (seconds)", fontsize=12)
plt.ylim(0, 55)
plt.yticks(np.arange(0, 56, 5))
plt.grid(True, linewidth=0.6, linestyle="--", alpha=0.7)
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

# Adjust layout and save
plt.tight_layout()
plt.savefig("simplified_algorithm_comparison_400.png", bbox_inches="tight", dpi=300)
plt.close()
