function(k)
    comp0 = ones(length(k))
    comp2 = k .^ 2 ./ 3       # k²/3 for sn2
    comp4 = k .^ 4 ./ 5       # k⁴/5 for sn4
    return hcat(comp0, comp2, comp4)
end
