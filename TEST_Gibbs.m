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
K1 = K1+eye(length(train_x))*eps;
[A,~] = chol(K1,'lower');

% plot conditional

gibbsz = train_z;
ISsamples = -10:0.01:10;
hold on
for gibbs_i = 1:1000
    for gibbsd = 1:train_n
        ISz = repmat(gibbsz,[1,length(ISsamples)]);
        ISz(gibbsd,:) = ISsamples;
        logw = Pz_Given_x(train_x,ISz,K1)+Ly_Given_z(ISz,train_t,para2);
        ind = IS('logweights',logw,'sample_n',1);
        gibbsz(gibbsd)=ISsamples(ind);
    end
    plot(gibbsz(1),gibbsz(2),'o')
    pause(eps)
end
return


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
options = optimoptions(@simulannealbnd,'FunctionTolerance',eps,'AnnealingFcn','annealingboltz');
u = randn(train_n,nSA);
zSA0 = A * u; % train_n-by-z_n
parfor i = 1:nSA
    [zSA(:,i),fvalSA(i)] = simulannealbnd(log_traget,zSA0(:,i),-10,10,options);
end
figure
plot(zSA(1,:),zSA(2,:),'.')
%% pick best of the bests
zSA = zSA(:,fvalSA==min(fvalSA));
zSA = repmat(zSA,[1,nSA]);
%% HM sample start from the peak

log_traget = @(z) Pz_Given_x(train_x,z,K1) + Ly_Given_z(z,train_t,para2);
log_traget_t = @(z) log_traget(z')';
z = MHSampling(log_traget_t,zSA','iter_n',1000,'adaptSig',0,'sig',0.2);
z = z';
figure
plot(z(1,:),z(2,:),'.')
%% prediction
pred_t = zeros(val_n,nSA);
pred_z = zeros(val_n,nSA);
parfor m = 1:nSA
    pred_z(:,m) = my_fitrgp(train_x,val_x,z(:,m),@kfcn,para1)';
    pred_t(:,m) = my_fitrgp(z(:,m),pred_z(:,m),train_t,@kfcn,para2)';
end
figure
hold on
for i = 1:nSA
    plot(val_x,pred_t(:,i),'.')
end
plot(val_x, val_t)
plot(val_x,mean(pred_t,2))