clear
close all
clc
load('data.mat')
if 0 % fake parameters
    para1 = [0,-2];
    papa2 = [0,-2];
end
addpath('..\matlab\Scripts_m\fun_MHSampling\')
addpath('..\matlab\Scripts_m\ex_PMCMC\')

%% note
% the problem is not identifiable
% because the prior and the likelihood are both semetric
% All the codes seems to be right
%% parameters
z_n=1000;% number of samples in anealing sampling (AS)
AS_n=1000000;% max number of cycles in AS
metro_n=100; % number of metropolis steps at each temperature
[~,A,w] = kfcn(1,1,para1)
[~,A,w] = kfcn(1,1,para2)
clear A W
temper_bounds=[0.2,1];
% bigger bounds gives smoother transition and thus better fit
% smaller bound gives fister convergence
%% training data
RAND_DATA = 1;
x_n = length(x);
if RAND_DATA
    train_n = 3;
    train_ind = sort(randsample(length(x),train_n));
else
    train_ind = 1:100:501;
    train_n = length(train_ind);
end
train_x = x(train_ind); % train_n-by-1
train_z = z(train_ind);
train_t = y(train_ind);

if 0
    val_n = 100;
    val_ind = sort(randsample(length(x),val_n));
else
    val_ind = 1:10:x_n;
    val_n = length(val_ind);
end
val_x = x(val_ind); % val_n-by-1
val_z = z(val_ind);
val_t = y(val_ind);
clear x y z

K1 = kfcn(train_x,train_x,para1);
K1 = K1+eye(length(train_x))*1e-10;
[A,~] = chol(K1,'lower');
%% plot the prior
if train_n == 2
    [mesh_X,mesh_Y] = meshgrid(-10:0.1:10);
    
    Z_prior = reshape(Pz_Given_x(train_x,[mesh_X(:),mesh_Y(:)]',K1),size(mesh_X));
    figure
    mesh(mesh_X,mesh_Y,exp(Z_prior))
    hidden off
    %% plot the likelihood
    Z_like = reshape(Ly_Given_z([mesh_X(:),mesh_Y(:)]',train_t,para2),size(mesh_X));
    figure
    mesh(mesh_X,mesh_Y,exp(Z_like))
    hidden off
    
    %% plot the posterior
    figure
    mesh(mesh_X,mesh_Y,exp(Z_prior + Z_like))
    hidden off
end

%% SA to find the peaks
log_traget = @(z) - sum( Pz_Given_x(train_x,z,K1) + Ly_Given_z(z,train_t,para2));

nSA = 70;
zSA = zeros(train_n,nSA);
fvalSA = zeros(1,nSA);
parfor i = 1:nSA
    zSA0 = randn(train_n,1);   % Starting point
    [zSA(:,i),fvalSA(i)] = simulannealbnd(log_traget,zSA0);
end

%% HM sample start from the peak
figure
plot(zSA(1,:),zSA(2,:),'.')
log_traget = @(z) Pz_Given_x(train_x,z,K1) + Ly_Given_z(z,train_t,para2);
log_traget_t = @(z) log_traget(z')';
z = MHSampling(log_traget_t,zSA','iter_n',10000,'adaptSig',0,'sig',0.1);
figure
plot(z(:,1),z(:,2),'.')