function myplot_tensors_2d_scattered(nu, XY, options)
% change to use color map for x axis.

% plot_tensors_2d - plot 2D tensor field
%
%   nu should have size (2,2,N)
%   XY should have size (N,2)
%
%   plot_tensors_2d_scattered(nu, XY, options);
%
%   options.nb_ellipses controls number of ellipses along each dimension.
%   options.image displays a background image.
%
%   Copyright (c) 2016 Gabriel Peyre


options.null = 0;
cmap = getoptions(options, 'cmap', 'parula');
fill_ellipses = getoptions(options, 'fill_ellipses', 1);
scaling = getoptions(options, 'scaling', .8);

p = size(nu,3);


% sort based on x-values
[sortedval sortidx] = sort(XY(:,1));

XY = XY(sortidx,:);
nu = nu(:,:,sortidx);


[e1,e2,l1,l2] = tensor_eigendecomp(nu);
theta = pi/2-atan2(e2(:,:,2), e2(:,:,1));
l_max = max(l1(:));
s = scaling; % scaling



switch cmap
    case 'jet'
        cols = jet(length(sortidx));
    case 'parula'
        cols = parula(length(sortidx));
end

hold on;
for k=1:p
    col = cols(k,:);
    ecol = col;
    if fill_ellipses
        ellipse_fill(s*l1(k)/l_max,s*l2(k)/l_max,theta(k),XY(k,1),XY(k,2),col, ecol);
    else
        ellipse(s*l1(k)/l_max,s*l2(k)/l_max,theta(k),XY(k,1),XY(k,2),col);
    end
end
axis tight; axis equal;
axis off; axis ij;

end

