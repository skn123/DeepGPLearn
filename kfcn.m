function Kmn = kfcn(Xm,Xn,theta) 
A=theta(1);
A=pos_bond(A,300); % upper limit
logW=theta(2);
[m,d]=size(Xm);
if d~=1
    error("need a Mx1 vector for Xm")
end
[n,d]=size(Xn);
if d~=1
    error("need a Mx1 vector for Xn")
end
Kmn=zeros(m,n);
for i=1:m
    for j=1:n
        dist=abs(Xm(i)-Xn(j))/exp(logW);
        Kmn(i,j)=A*exp(-dist^2/(2));
    end
end
end

