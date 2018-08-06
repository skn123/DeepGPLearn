function out=pos_bond(x,a)
out = a./(1+100.*exp(-(10/a)*x));