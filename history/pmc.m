function [X,W,Z] = pmc(log_target,dim,varargin)
%% ~
% Title:        Population Monte Carlo
% Authors:      Yousef El-Laham
% Description:  This function runs an adaptive importance sampling
%               algorithm called population Monte Carlo. The goal of the
%               methodology is to estimate some target distribution by
%               drawing a single from a population of proposal distribution.
%               The location parameters of the proposal are adapted via a
%               resampling step at each iteration of the algorithm. For the
%               purpose of simplicity, we use a normal distribution
%               as the proposal distribution.
% Inputs:       *Required Inputs*
%               ------------------
%               log_target -        log of the target distribution
%               dim -               dimension of the target distribution
%               
%               *Optional Inputs*
%               ------------------
%               NumProposals -      specify the number of proposals
%               NumSamples -        number of samples per proposal
%               NumIterations -     number of iteratios in algorithm
%               NumMixtures -       number of mixtures (if using partial DM
%                                   weighting scheme)
%               LocationInit -      option to initialize location parameter
%               ScaleInit -         option to initialize scale parameter
%               WeightingScheme -   Option to choose a weighting scheme:
%                                   1) Standard weights
%                                   2) Deterministic mixture (DM) weights
%                                   3) Partial DM weights
%               ResamplingScheme -  Option to choose resampling scheme:
%                                   1) Global resampling
%                                   2) Local resampling
% Outputs:      X -                 complete set of samples
%               W -                 complete set of weights
%               Z -                 normalizing constant
%% Assertions and Default Values for Optional Arguments

% Specific argument checks and error messages
isafunction= @(x) assert(isa(x,'function_handle'),...
    'This input must be of type "function handle"');
isaposinteger = @(x) assert((x>0)&&isnumeric(x)&&(round(x)==x),...
    'This input must be a positive integer');

% Assertions on the required inputs
isafunction(log_target);
isaposinteger(dim);

% Default values for optional input arguments
default_num_proposals=dim*10;
default_num_samples=5;
default_num_iterations=100;
default_num_mixtures=5;
default_mu_init=rand(default_num_proposals,dim);
default_sigma_init=eye(dim);
default_weighting_scheme='standard';
default_resampling_scheme='global';

%% Input parsing

% Define a new parser for the optional parameters
q=inputParser;
addParameter(q,'NumProposals',default_num_proposals,isaposinteger);
addParameter(q,'NumSamples',default_num_samples,isaposinteger);
addParameter(q,'NumIterations',default_num_iterations,isaposinteger);
addParameter(q,'NumMixtures',default_num_mixtures,isaposinteger);
addParameter(q,'LocationInit',default_mu_init);
addParameter(q,'ScaleInit',default_sigma_init);
addParameter(q,'WeightingScheme',default_weighting_scheme,@ischar);
addParameter(q,'ResamplingScheme',default_resampling_scheme,@ischar);
parse(q,varargin{:});
M=q.Results.NumProposals;
N=q.Results.NumSamples;
I=q.Results.NumIterations;
D=q.Results.NumMixtures;
mu_init=q.Results.LocationInit;
sigma_init=q.Results.ScaleInit;
WeightingScheme=q.Results.WeightingScheme;
ResamplingScheme=q.Results.ResamplingScheme;

% Check to see if location parameter configuration is the correct size
fprintf('Using...\n');
mu_rows=length(mu_init(:,1));
if(M~=mu_rows || isequal(mu_init,default_mu_init))
    fprintf('\t- Default location initialization\n');
    mu_init=rand(M,dim);
else
    fprintf('\t- Customized location initialization\n');
end
% Warn the user that a default scale initialization is being used
if(isequal(sigma_init,default_sigma_init))
    fprintf('\t- Default scale initialization\n');
else
    fprintf('\t- Customized scale initialization\n');
end
switch(WeightingScheme) % Determine q_x depending on weighting scheme
    case 'standard'
        fprintf('\t- Standard weights\n');
    case 'DM'
         fprintf('\t- DM weights\n');
    case 'partialDM'
        fprintf('\t- Partial DM weights\n');
end
switch(ResamplingScheme)
    case 'global'
        fprintf('\t- Global resampling\n');
    case 'local' 
        fprintf('\t- Local resampling\n');
end
fprintf('\n');

% Assertion functions for optional inputs
valid_location =@(x) assert(logical(prod(size(x)==[M,dim])),...
    'Location initialization is invalid');
valid_scale=@(x) assert(logical(prod(size(x)==[dim,dim])) && det(x)>0,...
    'Scale initialization is invalid');
valid_mixture_size=@(x) assert(x>=1 && x<=M && (round(x)==x),...
    'Setting for number of partial mixtures is invalid');

% Assertions for specified arguments
valid_location(mu_init);
valid_scale(sigma_init);
valid_mixture_size(D);

%% Algorithm

% Initialize empty arrays
X=[]; log_W=[];
% Initialize proposal parameters
mu=mu_init; mu_rep=repelem(mu,N,1);
sigma=sigma_init;
% Want to keep track of the largest log weight -> initialize it to -inf
max_log_w=-inf;
% Want to keep track which distr each sample came from
tag=repelem((1:M)',N);
% Over I iterations...
for i=1:I
    % Step 1: Draw samples from proposal distribution
    x=mvnrnd(mu_rep,sigma); X=[X;x];
    
    % Step 2: Weighting
    switch(WeightingScheme) % Determine q_x depending on weighting scheme
        case 'standard' % Standard weighting scheme
            q_x=mvnpdf(x,mu_rep,sigma);
        case 'DM' % Deterministic mixture weighting scheme
            obj = gmdistribution(mu,sigma);
            q_x=pdf(obj,x);
        case 'partialDM'
            distr_index=datasample((1:D)',M); % draw random mixture indices
            % loop over all partial mixtures
            q_x=zeros(M*N,1);
            for d=1:D
                mu_temp=mu(distr_index==d,:); % random mixture assignment
                obj = gmdistribution(mu_temp,sigma); % create a GMM
                % Which samples are assosciated with these means?
                samp_index=repelem(distr_index==d,N);
                % Evaluate q_x based on these samples
                q_x(samp_index)=pdf(obj,x(samp_index,:));
            end
        otherwise % by default use standard weights
            q_x=mvnpdf(x,mu_rep,sigma);
    end
    % Compute log weight
    log_w=log_target(x)-log(q_x);
    % Store the log weights at each iterations
    log_W=[log_W; log_w];
    % Update the maximum log weight if necessary
    if(max_log_w<max(log_w))
        max_log_w=max(log_w);
    end
    w=exp(log_w-max(log_w))+eps(0); % add very small number to avoid numerical errors
    
    % Step 3: Generate indicies of strongest particles (global or local
    % resampling)
    switch(ResamplingScheme)
        case 'global' % Global resampling
            w_norm=w./sum(w); % globalized softmax transformation
            index=datasample(1:M*N,M,'weights',w_norm);
        case 'local' % Local resampling
            w_reshape=reshape(w,N,M);
            normalize=repmat(sum(w_reshape,1),N,1); % partition function for each row
            w_norm=w_reshape./normalize;
            % Trick to do row-wise multinomial resampling
            index=mnrnd(1,w_norm').*repmat(1:N,M,1);
            % Convert row-wise indicies to particle indicies
            index=sum(index,2)+(0:N:(M*N-N))';
        otherwise % Global resampling by default
            w_norm=w./sum(w);
            index=datasample(1:M,M,'weights',w_norm);
    end
    
    % Step 4: Update proposal parameters using indicies
    mu=x(index,:); mu_rep=repelem(mu,N,1);
end
% Output the unnormalized set of importance weights
W=exp(log_W-max_log_w);
% IS estimate of normalizing constant 
Z=mean(W).*(exp(max_log_w));
end

