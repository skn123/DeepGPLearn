%% 2 stage GP with system built in function
%% This file just try to sample from z, currently no optimization on paras
% running
% but AS part is not resonable
clear
close all
clc
%% display controls
% display the histogram of sampled z(s)
show_hist=1;
%% parameters
N=10;% number of training data
M=1000;% number of particles in anealing sampling (AS)
nAS=1000;% max number of cycles in AS
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
if show_hist
  figure
  w=logw2w(logW);
  z=resample(z,w);
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
%% PMC
log_target_now=@(z)log_target(xt,z,tt,K1,para2); % log target
% Necessities
dim=N; % dimension of the desired target
Mpmc=1000; Npmc=10;% number of proposals and samples/proposal
Ipmc=2*10^6/(Mpmc*Npmc); % total number of iterations

% Optional initializations
Dpmc=10; % number of partial mixtures

% Run PMC
tic
[X,W,Z]=pmc(log_target_now,dim,...
  'NumProposals',Mpmc,...
  'NumSamples',Npmc,...
  'NumIterations',Ipmc,...
  'NumMixtures',Dpmc,...
  'WeightingScheme','partialDM',...
  'ResamplingScheme','global');
W_tilde=W./sum(W);
toc
%% visualization of the distribution of z
if show_hist
  figure
  posterior_samples=datasample(X,10000,'Weights',W_tilde);
  z=posterior_samples';
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
posterior_samples=datasample(X,NewM,'Weights',W_tilde);
z=posterior_samples';
M=NewM;
w=ones([M,1])/M;
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