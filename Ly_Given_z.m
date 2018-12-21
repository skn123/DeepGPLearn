function logW2 = Ly_Given_z(z,tt,para2)
[M,N] =size(z);
logW2=zeros([1,N]);
for i=1:N
  z_c=z(:,i);
  K2=kfcn(z_c,z_c,para2);
%   K2=0.5*(K2+K2');
  K2=K2+eye(M)*1e-10;
  logW2(i) = logmvnpdf(tt,tt*0,K2);
end