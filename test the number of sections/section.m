function s = section(d,L)
if d == 2
    s = L*2;
    return
end
for l = 1:L
    if l == 1
        s = 2;
    else
        s = s + section(d-1,l-1);
    end
end
