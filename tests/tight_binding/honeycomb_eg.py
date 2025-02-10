import numpy as np
import matplotlib.pyplot as plt

a = 1.0
t = 1.0
b1 = (2 * np.pi / a) * np.array([1, 1 / np.sqrt(3)])
b2 = (2 * np.pi / a) * np.array([0, 2 / np.sqrt(3)])
delta1 = (a / 2) * np.array([1, np.sqrt(3)])
delta2 = (a / 2) * np.array([1, -np.sqrt(3)])
delta3 = a * np.array([-1, 0])

def f_k(kx, ky):
    return -t * (np.exp(1j * (kx * delta1[0] + ky * delta1[1])) +
                 np.exp(1j * (kx * delta2[0] + ky * delta2[1])) +
                 np.exp(1j * (kx * delta3[0] + ky * delta3[1])))

kx_vals = np.linspace(-2*np.pi/a, 2*np.pi/a, 100)
ky_vals = np.linspace(-2*np.pi/a, 2*np.pi/a, 100)
kx, ky = np.meshgrid(kx_vals, ky_vals)
fk = f_k(kx, ky)

E1 = np.abs(fk)
E2 = -np.abs(fk)

fig, ax = plt.subplots(figsize=(8,6))
contour = ax.contourf(kx, ky, E1, levels=50, cmap='plasma')
plt.colorbar(contour, label="Energy")
ax.set_title("Honeycomb Lattice Band Structure")
ax.set_xlabel("$k_x$")
ax.set_ylabel("$k_y$")
plt.show()
