(input, output, D, Pkemu) -> output .* (exp(input[2])*1e-10 .* D^2) .^ 2
