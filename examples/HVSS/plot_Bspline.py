import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.interpolate import BSpline

plt.style.use("ggplot")
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
rcParams["font.size"] = 15
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
rcParams["text.usetex"] = True
rcParams["legend.numpoints"] = 1
rcParams["xtick.labelsize"] = 11
rcParams["ytick.labelsize"] = 11
rcParams["xtick.major.size"] = 7
rcParams["xtick.major.width"] = 1
rcParams["ytick.major.size"] = 7
rcParams["ytick.major.width"] = 1

path = "./spline"
# Parameters
degree = 3  # Degree of the B-spline (cubic)
num_basis = 5  # Number of basis functions to plot

# Construct a knot vector with uniform spacing
# The total number of knots is num_basis + degree + 1
knot_vector = np.linspace(0, 1, num_basis + degree + 1)

# Create a fine grid over the domain to evaluate the basis functions
x_vals = np.linspace(knot_vector[0], knot_vector[-1], 1000)

# Plot each basis function
plt.figure(figsize=(8, 6))
for i in range(num_basis):
    # Extract the knots for the i-th basis function
    t = knot_vector[i : i + degree + 2]
    # Create the basis function
    b_spline = BSpline.basis_element(t, extrapolate=False)
    # Evaluate the basis function over the grid
    y_vals = b_spline(x_vals)
    # Plot the basis function
    plt.plot(x_vals, y_vals, label=rf"$B_{{{i+1}}}^{{{degree}}}(u)$")

plt.plot(knot_vector, np.zeros(len(knot_vector)), "k*", mfc="none", ms=10, label=rf"knots")
plt.title(f"First {num_basis} B-spline Basis Functions (Degree {degree})")
plt.xlabel(r"$u$")
plt.ylabel("Basis Function Value")
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig(f"{path}/plots/B-splines.pdf", dpi=1000)
