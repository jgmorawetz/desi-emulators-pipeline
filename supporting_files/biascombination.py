import jax.numpy as jnp


def BiasCombination(biases):
    """
    Compute the bias combination coefficients for VelocILEPTors.
    This function takes 11 bias parameters and returns 19 coefficients:
    - 12 bias combination terms (b11 + bloop)
    - 4 counterterm coefficients (alpha0, alpha2, alpha4, alpha6)
    - 3 stochastic terms (sn, sn2, sn4)
    Args:
        biases: Array of 11 bias parameters [b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4]
    Returns:
        Array of 19 coefficients
    """
    b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4 = biases
    bias_coeffs = jnp.array(
        [
            1,
            b1,
            b1**2,  # b11 terms
            b2,
            b1 * b2,
            b2**2,  # bloop terms (b2)
            bs,
            b1 * bs,
            b2 * bs,
            bs**2,  # bloop terms (bs)
            b3,
            b1 * b3,  # bloop terms (b3)
            alpha0,
            alpha2,
            alpha4,
            alpha6,  # counterterms
            sn,
            sn2,
            sn4,  # stochastic terms
        ]
    )
    return bias_coeffs
