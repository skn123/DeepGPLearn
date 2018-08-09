%% 2 stage GP with system built in function
% Fixed annealing error, atleast working for small N
clear
close all
clc
%% display controls
% display the histogram of sampled z(s)
show_hist=0;
% display ESS during tempering process
show_info=0;
%% parameters
nTr=2;% number of training data
nSample=1000;% number of particles in anealing sampling (AS)
nAS=1000000;% max number of cycles in AS
nMetro=10; % number of metropolis steps at each temperature
para1=[1,2];% initial parameter
para2=[1,2];% initial parameter
temper_bounds=[0.5,1];
% bigger bounds gives smoother transition and thus better fit
% smaller bound gives fister convergence
%% training data
% xt=rand(nTr,1)*10;
% xt=sort(xt);
xt=(1:nTr);
xt=xt(:);
tt=sin(3*xt);
%% Sample from prior
K1=kfcn(xt,xt,para1);
K1=K1+eye(length(xt))*1e-5;
[E,~] = chol(K1);
u=randn(nTr,nSample);
z=E*u;
clear E u
%% AS
disp('start code generation')
tic
codegen AnnealedSampling -args {xt,z,tt,K1,para2,1,1,temper_bounds}
toc
disp('finish')

disp('start Annealed Sampling')
tic
z = AnnealedSampling_mex(xt,z,tt,K1,para2,nMetro,nAS,temper_bounds);
toc
disp('finish')
%% display B4 downsampling
figure
mvhist(z,5)
figure
plotRealizations(xt,z)
%% down sample
NewM=100;
if nSample>( NewM)
  ind=datasample(1:nSample,NewM,'Replace',false);
  z=z(:,ind);
  nSample=NewM;
end
clear NewM ind
%% test data
xVal=-5:0.01:15;
xVal=sort(xVal(:));
%% predict
nVal=length(xVal);
yVal=zeros(nVal,nSample);
parfor m=1:nSample
  zVal = my_fitrgp(xt,xVal,z(:,m),@kfcn,para1);
  yVal(:,m) = my_fitrgp(z(:,m),zVal,tt,@kfcn,para2);
end
clear nVal
%% disp
yPred=yVal*ones([nSample,1])/nSample;
ySD=sqrt(var(yVal,1,2));

figure
plot(xt,tt,'*')
hold on
plotRealizations(xVal,yVal)
title('prediction')