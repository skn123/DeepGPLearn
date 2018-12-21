function z = AnnealedSampling(xt,z,tt,K1,para2,nMetro,nAS,temper_bounds)
temper=0;
noiseSize=1;
for iAS=1:nAS
  % find log weights
  logW=log_target(xt,z,tt,nan,para2,1);
  % find temper difference from last one
  [w_tempered,temper_diff]=temper_weights(logW,temper_bounds);
  if (temper + temper_diff) >1
    temper_diff = 1 - temper;
    temper = 1;
  else
    temper = temper + temper_diff;
  end
  % Metropolis random walk
  logLh=@(z)log_target(xt,z,tt,K1,para2,temper);
  ESS_now=ESS(w_tempered);
  if temper == 1
    nMetro=max(500,nMetro*10);
  end
  for iMetro = 1:nMetro
    zp = z + randn(size(z)) * noiseSize;
    alpha=exp(logLh(zp)-logLh(z));
    u=rand(size(alpha));
    pick=u<alpha;
    z(:,pick)=zp(:,pick);
%     clc
%     info=['iAS: ',num2str(iAS),newline];
%     info=[info,'iMetro: ',num2str(iMetro),newline];
%     info=[info,'temper: ',num2str(temper),newline];
%     info=[info,'temper_diff: ',num2str(temper_diff),newline];
%     info=[info,'ESS_now: ',num2str(ESS_now),newline];
%     info=[info,'picked: ',num2str(sum(pick)),newline];
%     info=[info,'noiseSize: ',num2str(noiseSize),newline];
%     disp(info)
    if sum(pick(:))/length(pick(:)) < 0.1
      noiseSize=noiseSize*0.95;
    else
      noiseSize=noiseSize*2;
    end
  end
  if temper == 1% if ended by temperature = 1
    break
  end
end