clear
close all
clc
load('data.mat')
addpath('..\matlab\Scripts_m\fun_MHSampling\')
addpath('..\matlab\Scripts_m\ex_PMCMC\')

%% note
% the problem is not identifiable
% because the prior and the likelihood are both semetric
% All the codes seems to be right
%% parameters
z_n=10000;% number of samples in anealing sampling (AS)
AS_n=1000000;% max number of cycles in AS
metro_n=500; % number of metropolis steps at each temperature
[~,A,W] = kfcn(1,1,para1)
[~,A,W] = kfcn(1,1,para2)
clear A W
temper_bounds=[0.2,1];
% bigger bounds gives smoother transition and thus better fit
% smaller bound gives fister convergence
%% training data
train_n = 2;
train_ind = randsample(length(x),train_n);
train_sigv = 0;
train_x = x(train_ind);
train_z = z(train_ind);
train_t = y(train_ind);
clear x y z train_ind
%% Sample from prior
K1 = kfcn(train_x,train_x,para1);
K1 = K1+eye(length(train_x))*1e-10;
[A,~] = chol(K1,'lower');
u = randn(train_n,z_n);
z = A * u;
figure
mvhist(z,5)
clear A u

if train_n == 2
    %% plot the prior
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
%% direct importance sample

logW = Ly_Given_z(z,train_t,para2);
W = logw2w(logW);
if ESS(W)>0.3
    train_z
    bestsample = z(:,W==max(W))
    
    ind = IS('weights',W);
    z = z(:,ind);
    figure
    mvhist(z,5)
    med = median(z,2)
    estimation = mean(z,2)
    
    z1 = z(1,:);
    [nz,xz] = hist(z);
    
    Ly_Given_z(train_z,train_t,para2)
    Ly_Given_z(bestsample,train_t,para2)
    Ly_Given_z(med,train_t,para2)
    Ly_Given_z(estimation,train_t,para2)
    
end
%% sequential IS-MH sample
% init
temper=0;
for AS_i = 1: AS_n
    % get the log likelihood of current parameter
    logW = Ly_Given_z(z,train_t,para2);
    % find the perfered temper accoording to ESS
    [w_tempered_diff,temper_diff]=temper_weights(logW,temper_bounds);
    if (temper + temper_diff) >1
        temper_diff = 1 - temper;
        temper = 1;
    else
        temper = temper + temper_diff;
    end
    temper % for debug and progress
    % IS
    ind = IS('weights',w_tempered_diff);
    z = z(:,ind);
    % use MH samples to randomize the sample (avoid degeneracy)
    log_traget = @(z) Pz_Given_x(train_x,z,K1) + Ly_Given_z(z,train_t,para2)*temper;
    log_traget_t = @(z) log_traget(z')';
    z = MHSampling(log_traget_t,z','iter_n',metro_n)';
    if temper == 1
        % if last run, run more iterations to assure convergence
        tic
        z = MHSampling(log_traget_t,z','iter_n',metro_n*2)';
        toc
        break
    end
end
figure
mvhist(z,5)
mean(z,2)
train_z
return