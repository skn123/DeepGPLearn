clear
close all
clc
%% data generation
% parameters
para1 = [0,1];
para2 = [0,-1];
% generation process
x = 0:0.01:10;
x = x(:);
Kx = kfcn(x,x,para1) ;
Kx = Kx + eye(length(x))*1e-10;
Ax = chol(Kx,'lower');
z = Ax * randn(size(x));
Kz = kfcn(z,z,para2);
clear A W
Kz = Kz + eye(length(x))*1e-10;
Az = chol(Kz,'lower');
y = Az * randn(size(z));
% visualization
subplot(2,2,1)
plot(x,z)
xlabel('x')
ylabel('z')
subplot(2,2,4)
plot(z,y)
xlabel('z')
ylabel('y')
subplot(2,2,2)
plot(y,z)
xlabel('y')
ylabel('z')
subplot(2,2,3)
plot(x,y)
xlabel('x')
ylabel('y')
% save
save('data.mat','para1','para2','x','z','y')
