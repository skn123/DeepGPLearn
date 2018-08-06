function p=logmvnpdf(x,mu,sig)
mu=mu(:);
N=length(mu);
if size(x,2)==N
  x=x';
end
p=-1/2*log(det(2*pi*sig))...
  -1/2*(x-mu)'/sig*(x-mu);
p=diag(p);
end