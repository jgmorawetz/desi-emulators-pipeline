(input, output, emu) -> output .* exp(input[1]) * 1e-10 .* exp(-2 * input[6])
