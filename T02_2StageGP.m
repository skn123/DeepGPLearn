%% 2 stage GP with system built in function
% Fixed annealing error, atleast working for small N
clear
close all
clc
%% display controls
% display the histogram of sampled z(s)
show_hist=1;
% display ESS during tempering process
show_ESS=1;
%% parameters
nTr=3;% number of training data
nSample=200;% number of particles in anealing sampling (AS)
nAS=1000;% max number of cycles in AS
nMetro=10; % number of metropolis steps at each temperature
para1=[1,2];% initial parameter
para2=[1,2];% initial parameter
%% method controls
% Metropolis noise and its cooling down
coolDown_speed = nAS*1000;
maxNoiseSize = 0.5;
noiseSize_list = maxNoiseSize*exp(-(0:(nAS-1))/(coolDown_speed/5));
figure
plot(noiseSize_list)

% Weight tempering method
% 1: Guided by ESS
% 2: linear ^ temper_speed
Temp_metho=1;

if Temp_metho == 2
  temper_speed = 4;
  temper_list=((0:nAS)/nAS).^temper_speed;
  temperDiff_list=diff(temper_list);
  temper_list(1)=[];% for better visual
  figure
  subplot(2,1,1)
  plot(temper_list)
  subplot(2,1,2)
  plot(temperDiff_list)
end
%% training data
xt=rand(nTr,1)*10;
xt=sort(xt);
tt=sin(3*xt);
%% Sample from prior
K1=kfcn(xt,xt,para1);
K1=K1+eye(length(xt))*1e-5;
[E,~] = chol(K1);
u=randn(nTr,nSample);
z=E*u;
clear E u
%% AS
temper=0;
AS_end_flag=0;
figure
for iAS=1:nAS
  message=['iAS: ',num2str(iAS),newline];
  % find log weights
  logW=zeros(nTr,1);
  parfor iSample=1:nSample
    z_current=z(:,iSample);
    K2=kfcn(z_current,z_current,para2);
    K2=0.5*(K2+K2');
    K2=K2+eye(length(xt))*1e-5;
    logW(iSample) = logmvnpdf(tt,tt*0,K2);
  end
  % find untempered weights
  w=logw2w(logW);
  message=[message,'ESS: ',num2str(ESS(w)),newline];
  % find temper difference from last one
  if Temp_metho == 1
    [~,temper_diff]=temper_weights(logW);
    if (temper + temper_diff) >1
      AS_end_flag=1;
    else
      w_tempered = logw2w(logW*temper_diff);
      temper = temper + temper_diff;
    end
  end
  if Temp_metho == 2
    temper = temper + temperDiff_list(iAS);
    w_tempered = logw2w(logW*temperDiff_list(iAS));
  end
  message=[message,'temper: ',num2str(temper),newline];
  message=[message,'tempered_ESS: ',num2str(ESS(w_tempered)),newline];
  if any([iAS==nAS,AS_end_flag])% if last iteration or ended by temperature = 1
    if show_ESS
      clc
      message
    end
    break % skip propose new particles
  end
  % Metropolis random walk
  likelihood=@(z)log_target(xt,z,tt,K1,para2,temper);
  for iMetro = 1:nMetro
    zp = z + randn(size(z)) * noiseSize_list(iAS);
    alpha=exp(likelihood(zp)-likelihood(z));
    u=rand(size(alpha));
    pick=u<alpha;
    z(:,pick)=zp(:,pick);
    if show_hist
      mvhist(z,5)
    end
    if show_ESS
      clc
      message
      pause(0.001)
    end
    message=[message,'picked: ',num2str(sum(pick)),newline];
  end
end
clear iAS
%% visualization of the distribution of z
if show_hist
  mvhist(z,5)
end
%% down sample
NewM=100;
ind=datasample(1:nSample,NewM,'Replace',false,'Weights',w);
z=z(:,ind);
w=w(ind);
w=w/sum(w);
nSample=NewM;
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
yPred=yVal*w(:);
figure
hold on
plot(xVal,yPred)
plot(xt,tt,'*')