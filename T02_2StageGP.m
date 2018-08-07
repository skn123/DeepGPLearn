%% 2 stage GP with system built in function
%% This file just try to sample from z, currently no optimization on paras
% Fixed annealing error, atleast working for small N
clear
close all
clc
%% display controls
% display the histogram of sampled z(s)
show_hist=1;
% display ESS during tempering process
show_ESS=1;
%% method controls
% particle propose method
% 1: by gaussian approximation
% 2: by uniform noise, maxNoiseSize*flip(linear).^coolDown_speed
% 3: by combination
RS_metho=3;
if RS_metho == 2
  coolDown_speed = 10;
  maxNoiseSize = 1;
end

% Weight tempering method
% 1: Guided by ESS
% 2: linear ^ temper_speed
Temp_metho=1;
if Temp_metho == 2
  temper_speed = 4;
end
%% parameters
N=10;% number of training data
M=1000;% number of particles in anealing sampling (AS)
nAS=200;% max number of cycles in AS
para1=[1,2];% initial parameter
para2=[1,2];% initial parameter
%% training data
xt=rand(N,1)*10;
xt=sort(xt);
tt=sin(3*xt);
%% Sample from prior
K1=kfcn(xt,xt,para1);
K1=K1+eye(length(xt))*1e-5;
[E,p] = chol(K1);
u=randn(N,M);
z=E*u;
%% AS
if Temp_metho == 1
  temper=0;
end
if Temp_metho == 2
  temper_list=((0:nAS)/nAS).^temper_speed;
  temper_list=diff(temper_list);
end
if RS_metho == 2
  noiseSize_list = maxNoiseSize*((nAS:-1:1)/nAS).^coolDown_speed;
end
for iAS=1:nAS
  % find log weights 
  logW=zeros(N,1);
  parfor m=1:M
    z_current=z(:,m);
    K2=kfcn(z_current,z_current,para2);
    K2=0.5*(K2+K2');
    K2=K2+eye(length(xt))*1e-5;
    logW(m) = logmvnpdf(tt,tt*0,K2);
  end
  clc
  % find untempered weights
  w=logw2w(logW);
  % find tempered weights
  if Temp_metho == 1
    [~,temper_diff]=temper_weights(logW);
    if (temper + temper_diff) >1
      w_tempered = logw2w(logW*(1-temper));
      z = resample(z,w);
      break
    else
      w_tempered = logw2w(logW*temper_diff);
      temper = temper + temper_diff;
    end
  end
  if Temp_metho == 2
    w_tempered = logw2w(logW*temper_list(iAS));
  end
  if show_ESS
    disp("ESS")
    ESS(w)
    disp("tempered_ESS")
    ESS(w_tempered)
  end
  % resample to get equal weights
  z = resample(z,w_tempered); % z ~ tempered posterior_{t-1}
  if iAS==nAS
    break
  end
  if RS_metho == 1
    GMModel = fitgmdist(z',10); % approximated tempered posterior_{t-1}
    z=random(GMModel,M)';
  end
  if RS_metho == 2
    % perturb by uniform weights
    z = z + 2*(rand(size(z))-0.5) * noiseSize_list(iAS);
  end
  if RS_metho == 3
    p=partitionrnd(M,M);
    p=p*0.1+eye(M)*0.9;
    z=z*p;
  end
end
%% visualization of the distribution of z
if show_hist
  D=min(5,N);
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
%% down sample
NewM=100;
ind=datasample(1:M,NewM,'Replace',false,'Weights',w);
z=z(:,ind);
w=w(ind);
w=w/sum(w);
M=NewM;
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
y_pred=yv*w(:);
figure
hold on
plot(xv,y_pred)
plot(xt,tt,'*')