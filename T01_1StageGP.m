%% 1 stage GP with system built in function that optimize the aprameters
% working well
clear
close all
clc
%% data
if 0
  % observation noise
  train_n=10;
  train_x=rand(train_n,1)*10;
  train_x=sort(train_x);
  train_sigv=0;
  train_t=sin(3*train_x)+randn(size(train_x))*train_sigv;
  val_x=-5:0.01:15;
  val_x=sort(val_x(:));
  val_t=sin(3*val_x);
  save('data.mat')
else
  load('data.mat')
end
%% regression
theta=[1;0];
[~,A,W]=kfcn(1,1,theta)% for parameter display
clear A W
[val_y,val_ysd] = my_fitrgp(train_x,val_x,train_t,@kfcn,theta);
%% figure
figure
plot(train_x,train_t,'*')
hold on
plotRealizations(val_x,val_y,val_ysd)
title('1 stage prediction')

