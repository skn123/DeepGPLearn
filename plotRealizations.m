function plotRealizations(x,y)
nSample=size(y,2);
yPred=y*ones([nSample,1])/nSample;
ySD=sqrt(var(y,1,2));

px=[x;flip(x)]; 
py=[yPred+ySD; flip(yPred-ySD)];

hold on
plot(x,yPred)
patch(px,py,1,'FaceColor','black','FaceAlpha',.2,'EdgeColor','none');
