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
M=10;
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
  %   theta=[10.558888527436583;-0.204524413227546];
  theta=[1;-1];
  pos_bond(theta(1),300)
  exp(theta(2))
  [yv,ysd] = my_fitrgp(xt,xv,tt,@kfcn,theta);
end
%% compare
tv=sin(3*xv);
rms(yv-tv)
%% figure
close all
figure
hold on
plot(xt,tt,'*')
plot(xv,yv)
plot(xv,yv+ysd)
plot(xv,yv-ysd)

