%% 1 stage GP with system built in function that optimize the aprameters
% working well
clear
close all
clc
%% data
M=100;
xt=rand(M,1)*10;
xt=sort(xt);
tt=sin(3*xt);
%% train
parameter=[100,0];% initial parameter
GPM = fitrgp(xt,tt,...
  'KernelFunction',@kfcn,...
  'KernelParameters',parameter,...
  'OptimizeHyperparameters','auto');
% GPM = fitrgp(xt,tt,...
%   'KernelFunction',@kfcn,...
%   'KernelParameters',parameter,...
%   'OptimizeHyperparameters','auto',...
%   'Optimizer','lbfgs');
%% test
xv=-5:0.01:15;
xv=sort(xv(:));
[yv,ysd] = predict(GPM,xv);
%% compare
tv=sin(3*xv);
rms(yv-tv)
%% figure
close all
figure
hold on
plot(xv,tv,'*')
plot(xv,yv)
plot(xv,yv+ysd)
plot(xv,yv-ysd)

