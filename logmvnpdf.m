function p=logmvnpdf(x,mu,sig)
x=x(:);
mu=mu(:);
p=-1/2*log(det(2*pi*sig))...
  -1/2*(x-mu)'/sig*(x-mu);
end