import jax.numpy as jnp


def postprocessing(input, Pl, D):
    return Pl * (jnp.exp(input[1]) * 1e-10 * D**2) ** 2
