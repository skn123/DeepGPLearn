clear
close all
clc

mu = [1;1];
sig = [1,0.7;0.7,2];

R = mvnrnd(mu,sig,1000);
subplot(2,1,1)
plot(R(:,1),R(:,2),'.')
[mux,sigx] = congau(mu,sig,[4,0],2)
R = normrnd(mux,sigx,[1,1000]);
subplot(2,1,2)
hist(R)