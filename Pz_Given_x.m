function logW1 = Pz_Given_x(xt,z,K1)
N=size(z,2);
logW1=zeros([1,N]);
for i=1:N
    z_c=z(:,i);
    logW1(i)=logmvnpdf(z_c,xt*0,K1);
end