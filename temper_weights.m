function [w_tempered,temper]=temper_weights(logW,bounds)
% find a reasonalable tempered weigth from log weights
% guided by ESS
temper=1;
logW=logW-max(logW);
for trial = 1:1000
  w_tempered=exp(logW*temper);
  w_tempered=w_tempered/sum(w_tempered);
  if ESS(w_tempered)<min(bounds(:))
    temper=temper/1.1;
    continue
  end
  if ESS(w_tempered)>max(bounds(:))
    temper=temper*1.05;
    continue
  end
  if false % if debug
    trial
    ESS(w)
    temper
    figure
    hist(w,1000)
  end
  break
end