function [mux,sigx] = congau(mu,sig,x,dim)
% conditional gaussian parameter calculation
% dim : int scaler
x2 = x;
mu2 = mu;
sig2 = sig;
x2(dim) = [];
mu2(dim) = [];
sig2(dim,:) = [];
sig2(:,dim) = [];
mu1 = mu(dim);
sig1 = sig(dim,dim);
sig12 = sig(dim,:); % row vector
sig12(dim) = [];

sig12invsig2 = sig12/sig2;
mux = mu1 + sig12invsig2 * (x2 - mu2);
sigx = sig1 - sig12invsig2*(sig12');