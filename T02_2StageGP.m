%% 2 stage GP with system built in function
% loop of functions:
% sample Zs
% E(z) by average the samples
% M(parameters) by built in functions
%% Running but result does not make sense
% possible reason samples are too sparse
%%
clear
close all
clc
N=10;
M=200;
R=3;
xt=rand(N,1)*10;
xt=sort(xt);
tt=sin(3*xt);

%
for m=1:M
  para1=[100,0];% initial parameter
  para2=[100,0];% initial parameter
  u=randn(size(xt));
  K1=kfcn(xt,xt,para1);
  K1=K1+eye(length(xt))*1e-5;
  [E,p] = chol(K1);
  if p==0
    u=randn(size(xt));
    z(:,m)=E*u;
  end
  K2=kfcn(z(:,m),z(:,m),para2);
  K2=0.5*(K2+K2');
  K2=K2+eye(length(xt))*1e-5;
  w(m) = logmvnpdf(tt,tt*0,K2);
end

w=w-max(w);
w=exp(w);
w=w/sum(w);
z=z*w';

for r=1:R
  GPM1=fitrgp(xt,z,...
    'KernelFunction',@kfcn,...
    'KernelParameters',para1,...
    'OptimizeHyperparameters','auto');
  para1=GPM1.KernelInformation.KernelParameters;
  GPM2=fitrgp(z,tt,...
    'KernelFunction',@kfcn,...
    'KernelParameters',para2,...
    'OptimizeHyperparameters','auto');
  para2=GPM2.KernelInformation.KernelParameters;
  xv=rand(N,1)*10;
  xv=sort(xv);
  figure
  plot(xv,predict(GPM2,predict(GPM1,xv)))
  pause
  close all
  for m=1:M
    u=randn(size(xt));
    K1=kfcn(xt,xt,para1);
    K1=K1+eye(length(xt))*1e-5;
    [E,p] = chol(K1);
    if p==0
      u=randn(size(xt));
      z(:,m)=E*u;
    end
    K2=kfcn(z(:,m),z(:,m),para2);
    K2=0.5*(K2+K2');
    K2=K2+eye(length(xt))*1e-5;
    w(m) = logmvnpdf(tt,tt*0,K2);
  end
  plot(xt,z)
  w=w-max(w);
  w=exp(w);
  w=w/sum(w);
  z=z*w';
  
  xv=rand(N,1)*10;
  xv=sort(xv);
  figure
  plot(xv,predict(GPM2,predict(GPM1,xv)))
  pause
  close all
end
