function [yv,ysd] = my_fitrgp(xt,xv,tt,kfcn,theta)
% GP without optimization of parameter
x = [xt;xv];
N = size(x,1);
C = kfcn(x,x,theta);
C = C + eye(N)*1E-5;
N1=size(xt,1);
C11=C(1:N1,1:N1);
C22=C(N1+1:end,N1+1:end);
C12=C(1:N1,N1+1:end);
C21=C(N1+1:end,1:N1);

C21_x_invC11=C21/C11;
yv=C21_x_invC11*(tt);
ycov=C22-C21_x_invC11*C12;
ysd=sqrt(diag(ycov));
end

