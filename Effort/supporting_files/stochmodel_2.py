import jax.numpy as jnp


def StochModel(k):
    """
    Compute stochastic components for the quadrupole (l=2).
    Args:
        k: Array of k-values (wavenumbers)
    Returns:
        Array of shape (len(k), 3) containing coefficients for [sn, sn2, sn4]:
        - Column 0: zeros (no sn contribution)
        - Column 1: 2k²/3 (coefficient for sn2)
        - Column 2: 4k⁴/7 (coefficient for sn4)
    """
    comp0 = jnp.zeros(len(k))
    comp2 = 2 * k**2 / 3
    comp4 = 4 * k**4 / 7
    return jnp.column_stack((comp0, comp2, comp4))
