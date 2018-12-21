function [yv,ysd] = my_fitrgp_mean(train_x,val_x,train_t,kfcn,theta,f)
% f(x) is mean function
% GP without optimization of parameter
x = [train_x;val_x];
N = size(x,1);
C = kfcn(x,x,theta);
C = C + eye(N)*1E-5;
N1=size(train_x,1);
C11=C(1:N1,1:N1);
C22=C(N1+1:end,N1+1:end);
C12=C(1:N1,N1+1:end);
C21=C(N1+1:end,1:N1);

C21_x_invC11=C21/C11;
yv=f(val_x) + C21_x_invC11*(train_t-f(train_x));
ycov=C22-C21_x_invC11*C12;
ysd=sqrt(diag(ycov));
end

