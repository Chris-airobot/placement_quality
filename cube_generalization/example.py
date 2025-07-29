import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# Example half-extents for a face
du = 0.05  # half-length along opening (binormal) axis
dv = 0.10  # half-length along the other in-plane axis
G_max = 0.08  # gripper max half-opening span

# Compute psi_max for aperture-fit
thetas = np.linspace(0, np.pi/2, 500)
spans = du * np.cos(thetas) + dv * np.sin(thetas)
valid = thetas[spans <= G_max]
psi_max = valid.max() if valid.size else 0.0

# Unit vector along opening axis at psi_max
opening_dir = np.array([np.cos(psi_max), np.sin(psi_max)])

# Finger contact points at ±G_max along opening_dir
p1 =  opening_dir * G_max
p2 = -opening_dir * G_max

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
# Draw face rectangle centered at origin (in u-v plane)
rect = Rectangle((-du, -dv), 2*du, 2*dv, fill=False, edgecolor='black', linewidth=2)
ax.add_patch(rect)

# Draw axes: opening (binormal) and tilt axis
ax.arrow(0, 0, du, 0, head_width=0.005, length_includes_head=True, color='blue', label='Opening axis')
ax.arrow(0, 0, 0, dv, head_width=0.005, length_includes_head=True, color='green', label='Tilt axis')

# Plot finger positions at psi_max
ax.add_patch(Circle((p1[0], p1[1]), 0.007, color='red', label='Fingers at ψ_max'))
ax.add_patch(Circle((p2[0], p2[1]), 0.007, color='red'))

# Plot dashed line for opening direction
line_x = np.array([-du*1.2, du*1.2])
line_y = np.tan(psi_max) * line_x
ax.plot(line_x, line_y, linestyle='--', color='gray', label=f'Open dir @ ψ≈{np.degrees(psi_max):.1f}°')

# Adjust plot
ax.set_xlim(-dv*1.2, dv*1.2)
ax.set_ylim(-dv*1.2, dv*1.2)
ax.set_aspect('equal')
ax.set_xlabel('Opening (binormal) axis')
ax.set_ylabel('Other in-plane axis')
ax.set_title('Aperture-Fit: Gripper Opening Direction')
ax.legend(loc='upper right')
plt.grid(True)
plt.show()
