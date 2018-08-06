%% 2 stage GP with system built in function
%% This file just try to sample from z, currently no optimization on paras
% running but result not as expected
% w too sparse, one element of w is dominating the weights
clear
close all
clc
%% parameters
N=10;
M=10000;
R=3;
%% training data
xt=rand(N,1)*10;
xt=sort(xt);
tt=sin(3*xt);
%% sample M particles
z=zeros(N,M);
w=zeros(N,1);
parfor m=1:M
  para1=[10,0];% initial parameter
  para2=[10,0];% initial parameter
  u=randn(size(xt));
  K1=kfcn(xt,xt,para1);
  K1=K1+eye(length(xt))*1e-5;
  [E,p] = chol(K1);
  if p==0
    u=randn(size(xt));
    z_current=E*u;
  end
  K2=kfcn(z_current,z_current,para2);
  K2=0.5*(K2+K2');
  K2=K2+eye(length(xt))*1e-5;
  z(:,m) = z_current;
  w(m) = logmvnpdf(tt,tt*0,K2);
end
%% normalize weights
w=w-max(w);
w=exp(w);
w=w/sum(w);
%% find expectiation
z_mean=z*w;