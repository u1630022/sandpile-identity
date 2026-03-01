import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('Solarize_Light2')
df = pd.read_csv('data.csv', skipinitialspace=True)
df_plot = df.replace(-1, np.nan)

# Interpolate NaN values for line plotting
df_interpolated = df_plot.copy()
df_interpolated['iterated_burning_1'] = df_interpolated['iterated_burning_1'].interpolate(limit_area='inside')
df_interpolated['iterated_burning_10'] = df_interpolated['iterated_burning_10'].interpolate(limit_area='inside')
df_interpolated['exponential_burning'] = df_interpolated['exponential_burning'].interpolate(limit_area='inside')
df_interpolated['difference'] = df_interpolated['difference'].interpolate(limit_area='inside')
df_interpolated['surface'] = df_interpolated['surface'].interpolate(limit_area='inside')
df_interpolated['my_method'] = df_interpolated['my_method'].interpolate(limit_area='inside')

# Plot the data
plt.figure(figsize=(10, 6))

# Plot lines with interpolated values
plt.plot(df_interpolated['size'], df_interpolated['iterated_burning_1'], label='Iterated Burning k=1', linestyle='-', color='blue')
plt.plot(df_interpolated['size'], df_interpolated['iterated_burning_10'], label='Iterated Burning k=10', linestyle='-', color='purple')
plt.plot(df_interpolated['size'], df_interpolated['exponential_burning'], label='Exponential Burning', linestyle='-', color='orange')
plt.plot(df_interpolated['size'], df_interpolated['difference'], label='Difference', linestyle='-', color='green')
plt.plot(df_interpolated['size'], df_interpolated['surface'], label='Surface', linestyle='-', color='black')
plt.plot(df_interpolated['size'], df_interpolated['my_method'], label='My Method', linestyle='-', color='red')

# Plot markers only where there is actual data
plt.scatter(df_plot['size'], df_plot['iterated_burning_1'], color='blue', label='_nolegend_')
plt.scatter(df_plot['size'], df_plot['iterated_burning_10'], color='purple', label='_nolegend_')
plt.scatter(df_plot['size'], df_plot['exponential_burning'], color='orange', label='_nolegend_')
plt.scatter(df_plot['size'], df_plot['difference'], color='green', label='_nolegend_')
plt.scatter(df_plot['size'], df_plot['surface'], color='black', label='_nolegend_')
plt.scatter(df_plot['size'], df_plot['my_method'], color='red', label='_nolegend_')

# Add title and labels
plt.title('Average Runtime of Identity Calculation Methods')
plt.xlabel('Grid Size')
plt.ylabel('Runtime (seconds)')
plt.legend()
plt.grid(True)

# Save the plot as an SVG file
plt.savefig('runtime_comparison.svg', format='svg', dpi=200, bbox_inches='tight')
plt.close()

print("Graph saved as 'runtime_comparison.svg'")

