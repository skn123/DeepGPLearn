%% 2 stage GP with system built in function
% Fixed annealing error, atleast working for small N
clear
close all
clc
%% display controls
% display the histogram of sampled z(s)
show_hist=0;
% display ESS during tempering process
show_ESS=1;
%% parameters
nTr=3;% number of training data
nSample=100;% number of particles in anealing sampling (AS)
nAS=1000;% max number of cycles in AS
para1=[1,2];% initial parameter
para2=[1,2];% initial parameter
%% method controls
% Metropolis noise and its cooling down
% 1: use a function of time
% 2: adaptive
Cool_metho = 2;
if Cool_metho == 1
  coolDown_speed = nAS*10;
  maxNoiseSize = 0.5;
  noiseSize_list = maxNoiseSize*exp(-(0:(nAS-1))/(coolDown_speed/5));
end

% Adaptive Metropolis cycles
adaptive_nMetro = 1;
if adaptive_nMetro
  ESS_th=0.5;
  nMetro=30; % max number of metropolis steps
else
  nMetro=10; % number of metropolis steps at each temperature
end

if Cool_metho == 2
  noiseSize=1;
end
% Weight tempering method
% 1: Guided by ESS
% 2: linear ^ temper_speed
Temp_metho=1;
if Temp_metho == 1
  % bigger bounds gives smoother transition and thus better fit
  % smaller bound gives fister convergence
  temper_bounds=[0.9,1];
end
if Temp_metho == 2
  temper_speed = 2;
  temper_list=((1:nAS)/nAS).^temper_speed;
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
ESS_start=ESS(logw2w(log_target(xt,z,tt,K1,para2,0)))
pause
if adaptive_nMetro
  ESS_th=ESS_start;
end
clear E u
%% AS
temper=0;
AS_end_flag=0;
if show_hist==1
  figure
end
for iAS=1:nAS
  % find log weights
  logW=log_target(xt,z,tt,nan,para2,1);
  % find untempered weights
  w=logw2w(logW);
  % find temper difference from last one
  if Temp_metho == 1
    [~,temper_diff]=temper_weights(logW,temper_bounds);
    if (temper + temper_diff) >1
      AS_end_flag=1;
      temper_diff = 1 - temper;
      temper = 1;
    else
      temper = temper + temper_diff;
    end
    temper_list(iAS)=temper;
  end
  if Temp_metho == 2
    temper = temper_list(iAS);
  end
  % Metropolis random walk
  logLh=@(z)log_target(xt,z,tt,K1,para2,temper);
  for iMetro = 1:nMetro
    if Cool_metho == 1
      noiseSize = noiseSize_list(iAS);
    end
    zp = z + randn(size(z)) * noiseSize;
    alpha=exp(logLh(zp)-logLh(z));
    u=rand(size(alpha));
    pick=u<alpha;
    z(:,pick)=zp(:,pick);
    
    ESS_now=ESS(logw2w(logLh(z)));
    if show_ESS
      clc
      message=['iAS: ',num2str(iAS),newline];
      message=[message,'iMetro: ',num2str(iMetro),newline];
      message=[message,'temper: ',num2str(temper),newline];
      message=[message,'ESS_now: ',num2str(ESS_now),newline];
      message=[message,'picked: ',num2str(sum(pick)),newline];
      message=[message,'noiseSize: ',num2str(noiseSize),newline];
      message
    end
    if show_hist
      mvhist(z,5)
      pause(0)
    end
    if Cool_metho == 2
      if sum(pick(:))/length(pick(:)) < 0.1
        noiseSize=noiseSize*0.95;
      else
        noiseSize=noiseSize*1.05;
      end
      noiseSize_list(iAS)=noiseSize;
    end
    if adaptive_nMetro
      if all([ESS_now>ESS_th ,not(any([iAS==nAS,AS_end_flag]))])
        break
      end
    end
  end
  if any([iAS==nAS,AS_end_flag])% if last iteration or ended by temperature = 1
    break % skip propose new particles
  end
end
clear iAS
%% visualization of the distribution of z
if show_hist
  mvhist(z,5)
end
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
figure
hold on
plot(xVal,yPred)
plot(xt,tt,'*')
title('prediction')

figure
plot(temper_list)
title('temperature')

figure
plot(log(noiseSize_list))
title('log(noise size)')

figure
mvhist(z,5)