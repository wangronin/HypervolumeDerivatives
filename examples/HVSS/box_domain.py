import jax
import jax.numpy as jnp


def union_box_constraint(x: jnp.ndarray, centers: jnp.ndarray, half_radii: jnp.ndarray) -> jnp.ndarray:
    # Compute center‐distances (euclidean) for each box i:
    diff_cent = x[jnp.newaxis, :] - centers  # (N, d)
    dist_cent = jnp.linalg.norm(diff_cent, axis=-1)  # (N,)
    # find index of nearest box
    i_nearest = jnp.argmin(dist_cent)  # scalar index
    # Extract nearest center and half‐radius
    c_near = centers[i_nearest, :]  # shape (d,)
    h_near = half_radii[i_nearest, :]  # shape (d,)
    # Compute per‐axis distances
    diff = x - c_near  # shape (d,)
    abs_diff = jnp.abs(diff)  # shape (d,)
    # Inside‐margin along each axis: how far inside boundary if inside
    inside_margin = h_near - abs_diff  # shape (d,)
    # Determine if *inside* on all axes (i.e., inside the box)
    is_inside = jnp.all(inside_margin >= 0.0)
    outside_comp = abs_diff - h_near  # shape (d,)
    # If outside_comp negative means axis inside but other axis maybe outside → but for outside logic we use max(0, …)
    outside_comp = jnp.maximum(outside_comp, 0.0)
    g_vec = jnp.where(is_inside, -inside_margin, +outside_comp)
    return g_vec


def test_box_domain():
    # Suppose we have 4 boxes in 3D
    centers = jnp.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 0.5], [2.0, -1.0]])
    half_radii = jnp.array([[0.5, 0.5], [0.3, 0.3], [0.4, 0.4], [0.6, 0.6]])

    # A test point
    x_test = jnp.array([0.1, 1], dtype=jnp.float32)
    g_value = union_box_constraint(x_test, centers, half_radii)
    print("g(x_test) =", g_value)

    # Compute gradient using jax.grad
    grad_g = jax.jacobian(union_box_constraint, argnums=0)
    print("Grad g(x_test) =", grad_g(x_test, centers, half_radii))

    hessian_g = jax.hessian(union_box_constraint, argnums=0)
    print("Hessian g(x_test) =", hessian_g(x_test, centers, half_radii))
