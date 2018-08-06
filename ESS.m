function out = ESS(W)
% Efective sample size
out=1/sum(W.^2)/length(W);
end

