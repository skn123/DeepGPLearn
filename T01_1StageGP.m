%% 1 stage GP with system built in function that optimize the aprameters
% working well
clear
close all
clc
%% methods control
% 1 optimize kernel by a system chosen method
% 2 optimize kernel by 'lbfgs'
% 3 no optimzation on parameter
metho=3;

% observation noise
sigv=0;
%% data
M=100;
xt=rand(M,1)*10;
xt=sort(xt);
tt=sin(3*xt)+randn(size(xt))*sigv;
%% train
parameter=[100,0];% initial parameter
if metho==1
  GPM = fitrgp(xt,tt,...
    'KernelFunction',@kfcn,...
    'KernelParameters',parameter,...
    'OptimizeHyperparameters','auto');
end
if metho==2
  GPM = fitrgp(xt,tt,...
    'KernelFunction',@kfcn,...
    'KernelParameters',parameter,...
    'OptimizeHyperparameters','auto',...
    'Optimizer','lbfgs');
end
%% test data
xv=-5:0.01:15;
xv=sort(xv(:));
%% test
if any(find(metho==[1,2]))
  [yv,ysd] = predict(GPM,xv);
end
if metho==3
  theta=[-27.035541024693067;-0.428530372580549];
  [yv,ysd] = my_fitrgp(xt,xv,tt,@kfcn,theta);
end
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

