function [w,temper]=temper_weights(logW)
% find a reasonalable tempered weigth from log weights
% guided by ESS
temper=1;
logW=logW-max(logW);
for trial = 1:1000
  w=exp(logW*temper);
  w=w/sum(w);
  if ESS(w)<0.1
    temper=temper/2;
    continue
  end
  if ESS(w)>0.5
    temper=temper*1.1;
    continue
  end
  if false % if debug
    trial
    ESS(w)
    log(temper)
    figure
    hist(w,1000)
  end
  break
end