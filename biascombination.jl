biases -> begin
    b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4 = biases
    Array([1, b1, b1^2, b2, b1 * b2, b2^2, bs, b1 * bs, b2 * bs, bs^2, b3, b1 * b3,
        alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4])
end
