% Test for Sinkhorn and Barycenrters on 2x2 matrices in a 1D domain.
% Modified from qot toolbox
clear
clc


name = 'split';
%name = 'multi-orient';
%name = 'iso-orient';
%name = 'cross-orient';
%name = 'dirac-pairs';
%name = 'dirac-pairs-smooth';

rep = ['results/interpolation-1d/' name '/'];


trM = @(x)squeeze( x(1,1,:,:)+x(2,2,:,:) );

% number of tensors in each measure
N = [1,1]*200;
N = [1,1]*60;
N = [1,1]*30;
% number of barycenters
m = 9; 
% for display purposes
options.nb_ellipses = 30;
if strcmp(name, 'dirac-pairs')
    N = [1,1]*25;
    options.nb_ellipses = 25;
end

options.aniso = .9;
mu = load_tensors_pair(name,N, options);


% normalize mu! 
s = 10;
mutilde = cell(size(mu));
for ii = 1:length(mu)
    muii = mu{ii};
    mutilde{ii} = spdsimplexnormalize(muii, s);
end
mu = mutilde;


% display
clf
plot_tensors_1d(mu, options);



%% === linear interplations ===
muL = {};
for k=1:m
	t = (k-1)/(m-1);    % TODO: correct inversion
	muL{k} = mu{1}*(1-t) + mu{2}*t;
end
clf;
plot_tensors_1d(muL, options);


%% === QOT ===

% parameter: mode of computation of logM/expM
global logexp_fast_mode;
logexp_fast_mode = 1;

% Ground cost
c = ground_cost(N,1);
% regularization
%epsilon = (.15)^2;  % large
%epsilon = (.04)^2;  % small
%epsilon = (.01)^2;  % small
epsilon = (.06)^2;  % medium


% fidelity, , regularization for KL is rho/epsilon
%rho = 0.1;  %medium
%rho = 0.5;
%rho = 1; 
%rho = 10;  %medium


rho = 0.036 *5;
%rho = 0.36;
%rho = 0.36 * 5;


lambda = ceil(rho/epsilon);

% Sinkhorn
options.niter = 5000; 
options.disp_rate = 10;
options.disp_rate = NaN;
options.tau = 1.8*epsilon/(rho+epsilon);  % prox step, use extrapolation to seed up
[gamma,u,v,err] = quantum_sinkhorn(mu{1},mu{2},c,epsilon,rho, options);


% Compute interpolation using an heuristic formula.
options.sparse_mult = 30;
nu = quantum_interp(gamma, mu, m, 1, options);
% display 1D evolution as ellipses
clf;
plot_tensors_1d(nu, options);



%% === Main: Mat Sinkhorn ===
lambda_MOT = epsilon;
maxiter_MOT = 30;
options.lambda = lambda_MOT;
options.maxiter = maxiter_MOT;
options.M = mu{1};
options.N = mu{2};
[Gamma, DistBM] = BlockMOT(c, options);

% interpolation
options.sparse_mult = 30;
nu = quantum_interp(Gamma, mu, m, 1, options);
% display 1D evolution as ellipses
clf;
plot_tensors_1d(nu, options);

