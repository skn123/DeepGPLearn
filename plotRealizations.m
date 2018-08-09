function plotRealizations(x,y,y_SD)
if nargin == 2
  y_SD=sqrt(var(y,1,2));
end
y_n=size(y,2);
y_mean=y*ones([y_n,1])/y_n;

patch_x=[x;flip(x)];
patch_y=[y_mean+y_SD; flip(y_mean-y_SD)];

hold on
plot(x,y_mean)
patch(patch_x,patch_y,1,'FaceColor','black','FaceAlpha',.2,'EdgeColor','none');