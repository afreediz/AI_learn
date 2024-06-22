import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

x = np.linspace(0, 5, 10)  # 100 points from -5 to 5
y = np.linspace(0, 5, 10)
x, y = np.meshgrid(x, y)  # Create a meshgrid from x and y
z = x+y  # Compute z values (example function)

# Create figure and axes
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot 3D surface
ax.plot_surface(x, y, z, cmap='viridis')

# Customize labels and title
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Surface Plot')

# Add a color bar which maps values to colors
fig.colorbar(ax.plot_surface(x, y, z, cmap='viridis'), shrink=0.5, aspect=5)

# Show plot
plt.show()
