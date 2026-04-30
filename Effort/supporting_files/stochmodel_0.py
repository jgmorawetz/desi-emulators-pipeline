import jax.numpy as jnp


def StochModel(k):
    """
    Compute stochastic components for the monopole (l=0).
    Args:
        k: Array of k-values (wavenumbers)
    Returns:
        Array of shape (len(k), 3) containing coefficients for [sn, sn2, sn4]:
        - Column 0: ones (constant shot noise sn)
        - Column 1: k²/3 (coefficient for sn2)
        - Column 2: k⁴/5 (coefficient for sn4)
    """
    comp0 = jnp.ones(len(k))
    comp2 = k**2 / 3
    comp4 = k**4 / 5
    return jnp.column_stack((comp0, comp2, comp4))
