clear 
clc

global logexp_fast_mode;
logexp_fast_mode = 4; % fast mex
rng('default');
rng(0);

% use random tensor fields

%%
% Main parameters.


if not(exist('names'))
    names = {'disk', 'star5'};
    %names = {'annulus', 'star4'};
end
if not(exist('dist_mode'))
    dist_mode = 'euclidean';
end
if not(exist('init')) % which cloud is used for initialization
    init = 1;
end
% loss being used 
if not(exist('gw_loss'))
    gw_loss = 'l2';
    sigma = .8;
end
if strcmp(gw_loss(1:2),'kl') 
    if not(isempty(gw_loss(3:end)))
        sigma = str2num(gw_loss(3:end))/10;
    end
    gw_loss = 'kl';
end
% embedding method
mdsmethod = 'classical';


tensor_name = 'dirac-pairs-smooth';
%tensor_name = 'multi-orient';

%% 
% rep creation

Nshape = length(names);
name = '';
for i=1:Nshape
	name = [name names{i}]; 
    if not(i==Nshape)
        name = [name '-'];
    end
end

lossstr = gw_loss;
if strcmp(gw_loss,'kl')==1
    lossstr = [gw_loss num2str(round(sigma*10))];
end



%%
% Other parameters.

% size of the input clouds
N0 = 100; 
% parameter for geodesic distances computations.
t_Varadhan = 1e-2;  % time step for Varadhan formula

options.gw_loss = gw_loss;
% switch from matrix distance to kernel
switch gw_loss
    case 'l2'
        Kembed = @(d)d;
        iKembed = @(K)K;
    case 'kl'
        if not(exist('sigma'))
            sigma = 1;
        end
        Kembed = @(d)exp(-(d/sigma).^2);
        iKembed = @(K)sqrt(max(-log(K),0))*sigma;
    otherwise 
        error('Unknorn loss.');
end

% Helpers.

dotp = @(x,y)sum(x(:).*y(:));
mynorm = @(x)norm(x(:));
normalize = @(x)x/sum(x(:));

%%
% Load shapes.

X = {}; F = {}; D = {}; Mu = {}; N = [];
for i=1:Nshape
    opt.upsampling = 10;
    opt.smooth_iter = 0; % control smoothness of the boundary.
    switch dist_mode
        case 'euclidean'
            opt.meshing_mode = 'farthestpoint'; % crude meshing
        case 'geodesic'
            opt.meshing_mode = 'mesh2d'; % adaptive meshing
            opt.Nbound = 500; % #boundary point
    end    
    [V{i},F{i}] = load_shape_triangulation(names{i}, N0*5, opt);    
    switch dist_mode
        case 'euclidean'
            % use Euclidean distance matrices
            Dfull{i} = distmat(V{i}');
        case 'geodesic'
            % use geodesic distance matrix
            Op = load_mesh_operators(V{i}',F{i});
            U = inv( full( diag(Op.AreaV) + t_Varadhan * Op.Delta ));
            U = -t_Varadhan*log(U);
            Dfull{i} = sqrt(max(U,0));
    end    
    % perform subsampling
    Isub{i} = 1:size(Dfull{i},1);
    if size(Dfull{i},1)>N0
        % using euclidean distances for subsampling seems better
        Isub{i} = perform_fartherpoint_subsampling_euclidean(V{i},N0,1);
    end
    X{i} = V{i}(Isub{i},:);
    D{i} = Dfull{i}(Isub{i},Isub{i});
    D{i} = D{i}/median(D{i}(:));
    % corresponding kernel
    K{i} = Kembed(D{i});
    %
    N(i) = size(X{i},1);
    Mu{i} = normalize(ones(N(i),1));
end

%% settings for Tensor GW
% generate marginals
s = 10;

opt.aniso = .06;
Mu_mot = load_tensors_pair(tensor_name, N(1), opt);
Mu_mot{1} = spdsimplexnormalize(Mu_mot{1},s);
Mu_mot{2} = spdsimplexnormalize(Mu_mot{2},s);


%%
% plot inputs
for i = 1 : 2
    clf;
    options.scaling = 0.15;
    myplot_tensors_2d_scattered(Mu_mot{i}, X{i}, options);
end

%% compute barycenter
d = 2;
epsilon = 0;
tlist = linspace(0.2,0.8,4);

for k=1:length(tlist)
    t = tlist(k);
    w = [1-t t];
    fprintf('Barycenter %d/%d \n', k, length(tlist));
    
    options.lambda = epsilon;
    BWBaryiter = 15;
    options.GWOTiter = linspace(10, 30, BWBaryiter); % inner iteration
    %options.GWOTiter = 30 * ones(BWBaryiter,1); % inner iteration
    options.BWBaryiter = BWBaryiter; % outer iteration
    options.method = 'CG';
    options.tolpcg = 1e-5;
    options.tolgradnorm = 1e-6;
    Xbars{k} = w(1) * Mu_mot{1} + w(2) * Mu_mot{2};
    [Dbar] = BlockGWBarycenter(D{1}, D{2}, w, d, Mu_mot{1}, Mu_mot{2}, Xbars{k}, options);
    DbaryList{k} = Dbar;
    % run MDS
    opts.niter = 100; % for smacof
    opts.verb = 1;
   	opts.X0 = X{1};
    fprintf('Rendering  %d/%d \n', k, length(tlist));
    [X1,s] = perform_mds(DbaryList{k},size(X{1},2), mdsmethod, opts);
    % display
    X1s{k} = X1;
	clf;
    optplot.scaling = 0.07;
end