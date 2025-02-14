import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the CSV file
df = pd.read_csv('matrix_performance_row.csv')

# Set up the general style parameters
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['axes.grid'] = True

# Create figure 1: Line plot showing time vs dimension for each processor count
plt.figure()
for proc in df['processors'].unique():
    data = df[df['processors'] == proc]
    plt.plot(data['dimension'], data['time'], marker='o', label=f'{proc} processors', linewidth=2)

plt.title('Row Wise Matrix Multiplication Performance by Matrix Dimension', pad=20)
plt.xlabel('Matrix Dimension')
plt.ylabel('Time (seconds)')
plt.ylim(0, 325)
# Set y-axis ticks every 25 units
plt.yticks(np.arange(0, 326, 25))
plt.grid(True, 'major', 'y', linewidth=0.8)
plt.grid(True, 'major', 'x', linewidth=0.8)
plt.legend()
plt.savefig('performance_by_dimension_row.png', bbox_inches='tight')
plt.close()

# Create figure 2: Bar plot grouped by dimension
plt.figure()
df_pivot = df.pivot(index='dimension', columns='processors', values='time')
df_pivot.plot(kind='bar', width=0.7)
plt.title('Row Wise Matrix Multiplication Performance Comparison', pad=20)
plt.xlabel('Matrix Dimension')
plt.ylabel('Time (seconds)')
plt.ylim(0, 325)
# Set y-axis ticks every 25 units
plt.yticks(np.arange(0, 326, 25))
plt.grid(True, 'major', 'y', linewidth=0.8)
plt.legend(title='Number of Processors')
plt.tight_layout()
plt.savefig('performance_comparison_bars_row.png')
plt.close()

# Create figure 3: Heatmap
plt.figure(figsize=(10, 6))
pivot_table = df.pivot(index='processors', columns='dimension', values='time')
sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='YlOrRd')
plt.title('Row Wise Matrix Multiplication Performance Heatmap', pad=20)
plt.xlabel('Matrix Dimension')
plt.ylabel('Number of Processors')
plt.tight_layout()
plt.savefig('performance_heatmap_row.png')
plt.close()

print("Visualizations have been created successfully!")