function w=logw2w(logW)
% find weights from log weigths
logW=logW-max(logW);
w=exp(logW);
w=w/sum(w);