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
RAND_DATA = 0;
x_n = length(x);
if RAND_DATA
    train_n = 2;
    train_ind = sort(randsample(length(x),train_n));
else
    train_ind = 1:100:1001;
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
%% train
gprMdl = fitrgp(train_x,train_t);

%% prediction
[predmean,predstd] = predict(gprMdl,val_x);
RMSE = rms(predmean - val_t);
%% figure
figure
hold on
plot(val_x, val_t)
plot(val_x,predmean)

patch_x=[val_x;flip(val_x)];
patch_y=[predmean+predstd; flip(predmean-predstd)];
patch(patch_x,patch_y,1,'FaceColor','black','FaceAlpha',.2,'EdgeColor','none');
