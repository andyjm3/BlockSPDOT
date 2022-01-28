%%
% Test for Sinkhorn on 2x2 matrices in a 2D domain.
clear
clc

%name = '2d-smooth-rand';
%name = '2d-iso-bump';
%name = '2d-mixt-bump';
%name = '2d-aniso-fields';
name = '2d-bump-donut';

rep = ['results/interpolation-2d/' name '/'];
[~,~] = mkdir(rep);

n = 10; % width of images
N = n*n; % #pixels
op = load_helpers(n);

opt.aniso = .06;
opt.aniso = .1;

C = load_tensors_pair(name, n, opt);

opt.diffus_tau = .08;
opt.diffus_t = 50;
if strcmp(name, '2d-bump-donut')
    for k=1:2
        [e1,e2,l1,l2] = tensor_eigendecomp(C{k});
        l1 = max(l1,.03);
        l2 = max(l2,.03);
        C{k} = tensor_eigenrecomp(e1,e2,l1,l2);        
    end
    opt.diffus_t = 20;
end

mu = {}; 
for k=1:2    
    mu{k} = reshape(C{k},[2 2 N]);
end

% normalize mu
for ii = 1:length(mu)
    muii = mu{ii};
    mutilde{ii} = spdsimplexnormalize(muii);
end
mu = mutilde;

n1 = 256; % upscaling for display
opt.laplacian = 'superbases';
opt.laplacian = 'fd';
opt.disp_tensors = 1;
opt.render = @(x)texture_lut(x, 'red-metal');
[F,Fr] = Myrendering_tensors_2d(mu,n1, [rep 'input'], opt);





%% QOT
% Compute the coupling using Sinkhorn. 


global logexp_fast_mode;
logexp_fast_mode = 1; % slow
logexp_fast_mode = 4; % fast mex

% Ground cost
c = ground_cost(n,2);
% regularization
epsilon = (.08)^2;  % medium
% fidelity
rho = 1;  %medium
rho = 0.64; %which gives lambda = 100;
% run sinkhorn
options.niter = 500; % ok for .05^2
options.disp_rate = NaN; % no display
options.tau = 1.8*epsilon/(rho+epsilon);  % prox step, use extrapolation to seed up
fprintf('Sinkhorn: ');
[gamma,u,v,err] = quantum_sinkhorn(mu{1},mu{2},c,epsilon,rho, options);

% interpolation
m = 5;
opt.sparse_mult = 50;
opt.disp_tensors = 1;
fprintf('Interpolating: ');
nu = quantum_interp(gamma, mu, m, 2, opt);

%%
% Display with a background texture.

[F,Fr] = Myrendering_tensors_2d(nu,n1, [rep 'interpol'], opt);
for k=1:m
    imwrite( Fr{k}, [rep 'interpol-render-' num2str(k) '.png'], 'png' );
end



%% Mat Sinkhorn

lambda_MOT = epsilon;
%if strcmp(name, '2d-bump-donut')
%    lambda_MOT = 0;
%end
maxiter_MOT = 30;
options.lambda = lambda_MOT;
options.maxiter = maxiter_MOT;
options.M = mu{1};
options.N = mu{2};
[Gamma, DistBM] = BlockMOT(c, options);

% interpolation
opt.sparse_mult = 50;
opt.disp_tensors = 1;
nu = quantum_interp(Gamma, mu, m, 2, opt);

%% 
% Display with a background texture.

[F,Fr] = Myrendering_tensors_2d(nu,n1, [rep 'BlockMOT-interpol'], opt);
for k=1:m
    imwrite( Fr{k}, [rep 'BlockMOT-'  'interpol-render-' num2str(k) '.png'], 'png' );
end

