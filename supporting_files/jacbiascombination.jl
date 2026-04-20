function (biases)
    b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4 = biases
    T = eltype(biases)
    # Row indices, column indices, and values for non-zero entries
    I = [2, 3, 5, 8, 12,              # col 1 (b1)
        4, 5, 6, 9,                  # col 2 (b2)
        7, 8, 9, 10,                 # col 3 (bs)
        11, 12,                      # col 4 (b3)
        13,                          # col 5 (alpha0)
        14,                          # col 6 (alpha2)
        15,                          # col 7 (alpha4)
        16,                          # col 8 (alpha6)
        17,                          # col 9 (sn)
        18,                          # col 10 (sn2)
        19]                          # col 11 (sn4)
    J = [1, 1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
        4, 4,
        5,
        6,
        7,
        8,
        9,
        10,
        11]
    V = [one(T), 2 * b1, b2, bs, b3,    # derivatives w.r.t. b1
        one(T), b1, 2 * b2, bs,        # derivatives w.r.t. b2
        one(T), b1, b2, 2 * bs,        # derivatives w.r.t. bs
        one(T), b1,                  # derivatives w.r.t. b3
        one(T),                      # derivative w.r.t. alpha0
        one(T),                      # derivative w.r.t. alpha2
        one(T),                      # derivative w.r.t. alpha4
        one(T),                      # derivative w.r.t. alpha6
        one(T),                      # derivative w.r.t. sn
        one(T),                      # derivative w.r.t. sn2
        one(T)]                      # derivative w.r.t. sn4
    return sparse(I, J, V, 19, 11)
end
