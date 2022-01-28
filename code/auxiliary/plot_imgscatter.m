function plot_imgscatter(X, images, ds)
% images is a cell array of size N, each element is nrows-by-ncols-by-3
% X is scatter of size (N, 2)



dx = ds;
dy = ds;

ms = 30;


col = X;
N = size(X,1);
for i=1:size(col,2)
    col(:,i) = rescale(col(:,i));
end
if size(col,2)==2
    col(:,3) = col(:,1)*0;
end


hold on;
for i=1:N
    x = X(i,1); 
    y = X(i,2);
    plot(x, y, '.', 'MarkerSize', ms, 'color', col(i,:));
    
    %overlay an image
    xmin = x-dx ; xmax = x+dx ;
    ymin = y-dy ; ymax = y+dy ;
    
    image([xmin xmax],[ymin ymax],images{i});
end





axis ij;
axis tight; axis equal;  axis off; axis ij;



end

