function out=pos_bond(x,A)
% bound a positive number x by soft bound A
out = A./(1+exp(-((x-0.5*A)*10/A)));