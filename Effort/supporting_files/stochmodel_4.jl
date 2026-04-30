function(k)
    comp0 = zeros(length(k))         # no sn contribution
    comp2 = zeros(length(k))         # no sn2 contribution
    comp4 = 8 .* k .^ 4 ./ 35       # 8k⁴/35 for sn4
    return hcat(comp0, comp2, comp4)
end
