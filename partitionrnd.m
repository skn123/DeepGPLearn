function p = partitionrnd(D,N)
u=exprnd(1,[D,N]);
p=u;
parfor i=1:N
  p(:,i)=u(:,i)/sum(u(:,i));
end
end

