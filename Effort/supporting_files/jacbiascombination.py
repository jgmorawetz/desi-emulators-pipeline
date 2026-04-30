import jax.numpy as jnp


def JacobianBiasCombination(biases):
    """
    Compute the Jacobian of the bias combination with respect to bias parameters.
    This function returns the analytical Jacobian matrix of the bias combination,
    which is significantly faster than automatic differentiation for this operation.
    Args:
        biases: Array of 11 bias parameters [b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4]
    Returns:
        Jacobian matrix of shape (19, 11) where:
        - Rows correspond to the 19 bias combination terms
        - Columns correspond to derivatives w.r.t. [b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4]
    The bias combination produces 19 terms:
    [1, b1, b1^2,
     b2, b1*b2, b2^2,
     bs, b1*bs, b2*bs, bs^2,
     b3, b1*b3,
     alpha0, alpha2, alpha4, alpha6,
     sn, sn2, sn4]
    """
    b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4 = biases

    # Initialize a 19x11 Jacobian matrix with all zeros
    J = jnp.zeros((19, 11))

    # Column 0: Derivatives with respect to b1
    J = J.at[1, 0].set(1.0)  # d(b1)/db1
    J = J.at[2, 0].set(2 * b1)  # d(b1^2)/db1
    J = J.at[4, 0].set(b2)  # d(b1*b2)/db1
    J = J.at[7, 0].set(bs)  # d(b1*bs)/db1
    J = J.at[11, 0].set(b3)  # d(b1*b3)/db1

    # Column 1: Derivatives with respect to b2
    J = J.at[3, 1].set(1.0)  # d(b2)/db2
    J = J.at[4, 1].set(b1)  # d(b1*b2)/db2
    J = J.at[5, 1].set(2 * b2)  # d(b2^2)/db2
    J = J.at[8, 1].set(bs)  # d(b2*bs)/db2

    # Column 2: Derivatives with respect to bs
    J = J.at[6, 2].set(1.0)  # d(bs)/dbs
    J = J.at[7, 2].set(b1)  # d(b1*bs)/dbs
    J = J.at[8, 2].set(b2)  # d(b2*bs)/dbs
    J = J.at[9, 2].set(2 * bs)  # d(bs^2)/dbs

    # Column 3: Derivatives with respect to b3
    J = J.at[10, 3].set(1.0)  # d(b3)/db3
    J = J.at[11, 3].set(b1)  # d(b1*b3)/db3

    # Column 4: Derivatives with respect to alpha0
    J = J.at[12, 4].set(1.0)  # d(alpha0)/dalpha0

    # Column 5: Derivatives with respect to alpha2
    J = J.at[13, 5].set(1.0)  # d(alpha2)/dalpha2

    # Column 6: Derivatives with respect to alpha4
    J = J.at[14, 6].set(1.0)  # d(alpha4)/dalpha4

    # Column 7: Derivatives with respect to alpha6
    J = J.at[15, 7].set(1.0)  # d(alpha6)/dalpha6

    # Column 8: Derivatives with respect to sn
    J = J.at[16, 8].set(1.0)  # d(sn)/dsn

    # Column 9: Derivatives with respect to sn2
    J = J.at[17, 9].set(1.0)  # d(sn2)/dsn2

    # Column 10: Derivatives with respect to sn4
    J = J.at[18, 10].set(1.0)  # d(sn4)/dsn4
    J = J.at[18, 10].set(1.0)  # d(sn4)/dsn4

    return J
