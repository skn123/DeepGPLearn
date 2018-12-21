function mvhist(z,nDim)
D=min(nDim,size(z,1));
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