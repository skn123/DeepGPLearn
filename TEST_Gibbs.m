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
    train_n = 10;
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

K1 = kfcn(train_x,train_x,para1);
K1 = K1+eye(length(train_x))*eps;
[A,~] = chol(K1,'lower');

% plot conditional
gibbsn = 1000;
gibbsburnin = 1000;
thinning = 10;
gibbsz = zeros(train_n,1); % initial
gibbssample = zeros(train_n,gibbsn); % memory allocation
step = 0.1;
hold on
for gibbs_i = 1:gibbsn + gibbsburnin
    for gibbsd = 1:train_n
        ISsamples = -3*sqrt(K1(gibbsd,gibbsd)):step:3*sqrt(K1(gibbsd,gibbsd));
        ISsamples = ISsamples + rand(size(ISsamples)) * step;
        ISz = repmat(gibbsz,[1,length(ISsamples)]);
        ISz(gibbsd,:) = ISsamples;
        logw = Pz_Given_x(train_x,ISz,K1)+Ly_Given_z(ISz,train_t,para2);
        ind = IS('logweights',logw,'sample_n',1);
        gibbsz(gibbsd)=ISsamples(ind);
    end
    if gibbs_i > gibbsburnin
        gibbssample(:,gibbs_i-gibbsburnin) = gibbsz;
    end
end
% plot(gibbssample(1,:),gibbssample(2,:),'.')
z = gibbssample(:,1:thinning:end);
gibbssample_n = length(z);
%% prediction
pred_t = zeros(val_n,gibbssample_n);
pred_z = zeros(val_n,gibbssample_n);
parfor m = 1:gibbssample_n
    pred_z(:,m) = my_fitrgp(train_x,val_x,z(:,m),@kfcn,para1)';
    pred_t(:,m) = my_fitrgp(z(:,m),pred_z(:,m),train_t,@kfcn,para2)';
end
figure
hold on
for i = 1:gibbssample_n
    plot(val_x,pred_t(:,i),'.')
end
plot(val_x, val_t)
plot(val_x,mean(pred_t,2))