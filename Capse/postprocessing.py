import jax.numpy as jnp

def postprocessing(input, Cl):
    return Cl * jnp.exp(input[0])*1e-10 * jnp.exp(-2 * input[5])
