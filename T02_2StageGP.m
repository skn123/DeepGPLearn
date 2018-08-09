%% 2 stage GP with system built in function
% Fixed annealing error, atleast working for small N
clear
close all
clc
%% parameters
z_n=1000;% number of samples in anealing sampling (AS)
AS_n=1000000;% max number of cycles in AS
metro_n=10; % number of metropolis steps at each temperature
para1=[1,0];% initial parameter
para2=[1,0];% initial parameter
[~,A,W]=kfcn(1,1,para1)
[~,A,W]=kfcn(1,1,para2)
clear A W
temper_bounds=[0.5,1];
% bigger bounds gives smoother transition and thus better fit
% smaller bound gives fister convergence
%% training data
if 0
  % observation noise
  train_sigv=0;
  train_n=10;
  train_x=rand(train_n,1)*10;
  train_x=sort(train_x);
  train_t=sin(3*train_x)+randn(size(train_x))*train_sigv;
  val_x=-5:0.01:15;
  val_x=sort(val_x(:));
  val_t=sin(3*val_x);
  save('data.mat')
else
  load('data.mat')
end
%% Sample from prior
K1=kfcn(train_x,train_x,para1);
K1=K1+eye(length(train_x))*1e-5;
[E,~] = chol(K1);
u=randn(train_n,z_n);
z=E*u;
clear E u
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
%% down sample
sample_n_new=100;
if z_n>( sample_n_new)
  ind=datasample(1:z_n,sample_n_new,'Replace',false);
  z=z(:,ind);
  z_n=sample_n_new;
end
clear sample_n_new ind
%% save result
timeMarker=datestr(datetime('now'),'yymmddHHMM');
save(['TwoStageGP',timeMarker,'.mat'])
%% predict
val_n=length(val_x);
val_y=zeros(val_n,z_n);
parfor m=1:z_n
  val_z = my_fitrgp(train_x,val_x,z(:,m),@kfcn,para1);
  val_y(:,m) = my_fitrgp(z(:,m),val_z,train_t,@kfcn,para2);
end
clear nVal
%% disp
figure
plot(train_x,train_t,'*')
hold on
plotRealizations(val_x,val_y)
title('prediction')