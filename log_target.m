function logW=log_target(xt,z,tt,K1,para2,temper)
N=size(z,2);
logW=zeros([1,N]);
for i=1:N
  z_c=z(:,i);
  % first stage
  logW1=logmvnpdf(z_c,xt*0,K1);
  % second stage
  K2=kfcn(z_c,z_c,para2);
  K2=0.5*(K2+K2');
  K2=K2+eye(length(xt))*1e-5;
  logW2 = logmvnpdf(tt,tt*0,K2);
  % together
  logW(i)=logW1+logW2*temper;
end