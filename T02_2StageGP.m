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
nTr=10;% number of training data
nSample=1000;% number of particles in anealing sampling (AS)
nAS=200;% max number of cycles in AS
para1=[1,2];% initial parameter
para2=[1,2];% initial parameter
%% method controls
% particle propose method
% 1: by gaussian mixture approximation
% 2: by uniform noise, maxNoiseSize*flip(linear).^coolDown_speed
% 3: Metropolis random walk
Prop_metho = 3;

if Prop_metho == 1
  nGMcomponent = 5;
end
if any(Prop_metho == [2,3])
  coolDown_speed = nAS*3;
  maxNoiseSize = 1;
  noiseSize_list = maxNoiseSize*exp(-(0:(nAS-1))/(coolDown_speed/5));
  %   noiseSize_list=noiseSize_list-noiseSize_list(end);
  %   noiseSize_list=noiseSize_list/noiseSize_list(1);
  figure
  plot(noiseSize_list)
end
if Prop_metho == 3
  nMetro=200;
end

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
  % find tempered weights
  if Temp_metho == 1
    [~,temper_diff]=temper_weights(logW);
    if (temper + temper_diff) >1
      AS_end_flag=1;
    else
      w_tempered = logw2w(logW*temper_diff);
      temper = temper + temper_diff;
    end
    clear temper_diff
  end
  if Temp_metho == 2
    temper = temper + temperDiff_list(iAS);
    w_tempered = logw2w(logW*temperDiff_list(iAS));
  end
  message=[message,'temper: ',num2str(temper),newline];
  message=[message,'tempered_ESS: ',num2str(ESS(w_tempered)),newline];
  if any([iAS==nAS,AS_end_flag])% if last iteration or ended by temperature = 1
    w_tempered = logw2w(logW*(1-temper));
    z = resample(z,w_tempered);
    if show_ESS
      clc
      message
    end
    break % skip propose new particles
  end
  % resample to get equal weights
  z = resample(z,w_tempered); % z ~ prior * tempered likelihood_{t-1}
  
  if Prop_metho == 1
    % gaussian mixture approximation
    GMModel = fitgmdist(z',nGMcomponent);
    z=random(GMModel,nSample)';
    clear GMModel
  end
  if Prop_metho == 2
    % perturb by uniform weights
    z = z + 2*(rand(size(z))-0.5) * noiseSize_list(iAS);
  end
  if Prop_metho == 3
    % Metropolis random walk
    likelihood=@(z)log_target(xt,z,tt,K1,para2,temper);
    for iMetro = 1:nMetro
      zp = z + randn(size(z)) * noiseSize_list(iAS);
      alpha=exp(likelihood(zp)-likelihood(z));
      u=rand(size(alpha));
      pick=u<alpha;
      z(:,pick)=zp(:,pick);
    end
    message=[message,'picked: ',num2str(sum(pick)),newline];
  end
  if show_ESS
    clc
    message
    if show_hist
      mvhist(z,5)
    end
    pause(0.001)
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