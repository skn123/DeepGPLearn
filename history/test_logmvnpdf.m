% test multiple input for the function 
% and measure the speed
clear
close all
clc

D=10;
N=100;
mu=randn(D,1);
sig=eye(D);
x=randn(D,N);
p=mu*0;
tic
for i=1:N
  p(i)=logmvnpdf(x(:,i),mu,sig);
end
toc

tic
p2=logmvnpdf(x,mu,sig);
toc
figure
hold on
plot(p)
plot(p2)