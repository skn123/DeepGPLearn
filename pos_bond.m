function out=pos_bond(x,A)
out = A./(1+exp(-((x-0.5*A)*10/A)));