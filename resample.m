function [z_RSed,ind]=resample(z,w)
% resample a vector z by weight w
N=length(w(:));
ind=datasample(1:N,N,'weights',w);
if size(z,1)==N
  z_RSed=z(ind,:);
else
  z_RSed=z(:,ind);
end