%% 2 stage GP with system built in function
%% This file just try to sample from z, currently no optimization on paras
% running and the result is reasonable
clear
close all
clc
%% display controls
% display the histogram of sampled z(s)
show_hist=0;
% display ESS during tempering process
show_ESS=1;
%% method controls
% particle propose method
% 1: by gaussian approximation
RS_metho=1;
% Weight tempering method
% 1: Guided by ESS
% 2: linear^a
Temp_metho=1;
%% parameters
N=30;% number of training data
M=1000;% number of particles in anealing sampling (AS)
nAS=200;% max number of cycles in AS
para1=[1,0];% initial parameter
para2=[1,0];% initial parameter
%% training data
xt=rand(N,1)*10;
xt=sort(xt);
tt=sin(3*xt);
%% Initial sample of the particles
K1=kfcn(xt,xt,para1);
K1=K1+eye(length(xt))*1e-5;
[E,p] = chol(K1);
z=zeros(N,M);
logW=zeros(N,1);
parfor m=1:M
  if p==0
    u=randn(size(xt));
    z_current=E*u;
  end
  K2=kfcn(z_current,z_current,para2);
  K2=0.5*(K2+K2');
  K2=K2+eye(length(xt))*1e-5;
  z(:,m) = z_current; % Samples
  logW(m) = logmvnpdf(tt,tt*0,K2); % log weights
end
%% AS
% initialize the tempered weights for the first gaussian approximation
w_tempered=temper_weights(logW);
for iAS=1:nAS
  % Sample from proposal
  if RS_metho == 1
    % find MVGaussian Approximation
    z_RSed=resample(z,w_tempered);
    GMModel = fitgmdist(z_RSed',1);
    z=random(GMModel,M);
    z=z';
  end
  logW1=logmvnpdf(z,xt*0,K1);
  logW2=zeros(N,1);
  parfor m=1:M
    z_current=z(:,m);
    K2=kfcn(z_current,z_current,para2);
    K2=0.5*(K2+K2');
    K2=K2+eye(length(xt))*1e-5;
    logW2(m) = logmvnpdf(tt,tt*0,K2);
  end
  logW=logW1+logW2;
  % get real weights
  w=logw2w(logW); 
  % get tempered weights
  if Temp_metho==1
    [w_tempered,temper]=temper_weights(logW);
  end
  if Temp_metho==2
    temper=(iAS/nAS)^3;
    logW=logW*temper;
    logW=logW-max(logW);
    w_tempered=exp(logW);
    w_tempered=w_tempered/sum(w_tempered);
  end
  if show_ESS
    disp("tempered_ESS")
    ESS(w_tempered)
    disp("ESS")
    ESS(w)
  end
  if ESS(w)>0.5
    break
  end
end
%% down sample
NewM=100;
ind=datasample(1:M,NewM,'Replace',false,'Weights',w);
z=z(:,ind);
w=w(ind);
w=w/sum(w);
M=NewM;
%% visualization of the distribution of z
if show_hist
  D=5;
  for i=1:D
    for j=1:D
      subplot(D,D,(j-1)*D+i)
      if i==j
        hist(z(i,:),100)
      else
        plot(z(i,:),z(j,:),'.')
      end
    end
  end
end
%% test data
xv=-5:0.01:15;
xv=sort(xv(:));
%% predict
Nv=length(xv);
yv=zeros(Nv,M);
parfor m=1:M
  z_sim = my_fitrgp(xt,xv,z(:,m),@kfcn,para1);
  yv(:,m) = my_fitrgp(z(:,m),z_sim,tt,@kfcn,para2);
end
%% disp
y_pred=yv*w;
figure
hold on
plot(xv,y_pred)
plot(xt,tt,'*')