clear
close all
clc
load('data.mat')
%% split data
%% parameters
z_n=1000;% number of samples in anealing sampling (AS)
AS_n=1000000;% max number of cycles in AS
metro_n=100; % number of metropolis steps at each temperature
[~,A,W] = kfcn(1,1,para1)
[~,A,W] = kfcn(1,1,para2)
clear A W
temper_bounds=[0.5,1];
% bigger bounds gives smoother transition and thus better fit
% smaller bound gives fister convergence
%% training data
train_n = 10;
train_ind = randsample(length(x),train_n);
train_sigv = 0;
train_x = x(train_ind);
train_z = z(train_ind);
train_t = y(train_ind);
clear x y z
%% Sample from prior
K1 = kfcn(train_x,train_x,para1);
K1 = K1+eye(length(train_x))*1e-10;
[A,~] = chol(K1,'lower');
u = randn(train_n,z_n);
z = A * u;
clear A u
%% AS
disp('start code generation')
tic
codegen AnnealedSampling -args {train_x,z,train_t,K1,para2,1,1,temper_bounds}
toc
disp('finish')

disp('start Annealed Sampling')
tic
z = AnnealedSampling_mex(train_x,z,train_t,K1,para2,metro_n,AS_n,temper_bounds);
toc
disp('finish')
%% display B4 downsampling
figure
mvhist(z,5)

mean(z,2)
train_z