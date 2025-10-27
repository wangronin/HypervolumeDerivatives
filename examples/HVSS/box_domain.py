import jax
import jax.numpy as jnp


def box_distance(x: jnp.ndarray, c: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
    """
    Compute distance from point `x` to the box defined by center `c` and half-radius `r`.

    If x is inside the box (i.e., |x−c| ≤ r elementwise), the distance is 0.
    If outside, we compute the Euclidean norm of how far x exceeds the box bounds.

    Args:
      x: (d,) array
      c: (d,) array, box center
      r: (d,) array, half‐radius along each dimension
    Returns:
      scalar ≥ 0: the (Euclidean) distance to the box.
    """
    # element‐wise difference from center
    delta = jnp.abs(x - c)
    # outside amount (how much it exceeds radius)
    excess = jnp.maximum(delta - r, 0.0)
    return excess


def union_box_constraint(x: jnp.ndarray, centers: jnp.ndarray, half_radii: jnp.ndarray) -> jnp.ndarray:
    """
    Compute g(x) for a union of boxes. If x is inside *any* box, return zero.
    Otherwise return min_k distance to box k.

    Args:
      x: (d,) array
      centers: (K, d) array of box centers
      half_radii: (K, d) array of half‐radii for each box
    Returns:
      scalar g(x)
    """
    # vectorised over boxes
    excess = jax.vmap(lambda c, r: box_distance(x, c, r))(centers, half_radii)
    # Check if inside any box — equivalent to zero distance for that box
    return jnp.min(excess, axis=0)


def test_box_domain():
    # Example usage:
    d = 3
    K = 4
    # Suppose we have 4 boxes in 3D
    centers = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-1.0, 0.5, 2.0], [2.0, -1.0, 0.0]])
    half_radii = jnp.array([[0.5, 0.5, 0.5], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6]])

    # A test point
    x_test = jnp.array([10, 0, 0], dtype=jnp.float32)
    g_value = union_box_constraint(x_test, centers, half_radii)
    print("g(x_test) =", g_value)

    # Compute gradient using jax.grad
    grad_g = jax.jacobian(union_box_constraint, argnums=0)
    print("grad g(x_test) =", grad_g(x_test, centers, half_radii))

    hessian_g = jax.hessian(union_box_constraint, argnums=0)
    print("grad g(x_test) =", hessian_g(x_test, centers, half_radii))
