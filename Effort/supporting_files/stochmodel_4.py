import jax.numpy as jnp


def StochModel(k):
    """
    Compute stochastic components for the hexadecapole (l=4).
    Args:
        k: Array of k-values (wavenumbers)
    Returns:
        Array of shape (len(k), 3) containing coefficients for [sn, sn2, sn4]:
        - Column 0: zeros (no sn contribution)
        - Column 1: zeros (no sn2 contribution)
        - Column 2: 8k⁴/35 (coefficient for sn4)
    """
    comp0 = jnp.zeros(len(k))
    comp2 = jnp.zeros(len(k))
    comp4 = 8 * k**4 / 35
    return jnp.column_stack((comp0, comp2, comp4))
