function(k)
    comp0 = zeros(length(k))         # no sn contribution
    comp2 = 2 .* k .^ 2 ./ 3        # 2k²/3 for sn2
    comp4 = 4 .* k .^ 4 ./ 7        # 4k⁴/7 for sn4
    return hcat(comp0, comp2, comp4)
end
