clear 
clc
rng('default');
rng(9);

global logexp_fast_mode;
logexp_fast_mode = 4; % fast mex

name = '2d-bary';

d = 2;
n = 4;
N = n*n; % #pixels
op = load_helpers(n);

rep = ['results/barycenters-2d/' name '-n' num2str(n) '/'];
if not(exist(rep))
    mkdir(rep);
end

C = load_tensors_pair(name, n);
mu = {}; 
for k=1:length(C)   
    mu{k} = reshape(C{k},[2 2 N]);
    mu{k} = spdsimplexnormalize(mu{k});
end

% Ground cost
x = linspace(0,1,n);
[y,x] = meshgrid(x,x);
[X1,X2] = meshgrid(x(:),x(:));
[Y1,Y2] = meshgrid(y(:),y(:));
c0 = (X1-X2).^2 + (Y1-Y2).^2;
resh = @(x)reshape( x, [2 2 N N]);
flat = @(x)reshape(x, [2 2 N N]);
c = resh( tensor_diag(c0(:),c0(:)) );


%col = [1 0 0; 0 1 0; 0 0 1; 1 1 0];
col = [0.6350 0.0780 0.1840; 0.4660 0.6740 0.1880; 0 0.4470 0.7410; 0.8500 0.3250 0.0980];


% ploting convergence
lw = 2.0;
ms = 5.0;
fs = 20;
colors = {[55, 126, 184]/255, [228, 26, 28]/255, [247, 129, 191]/255, ...
          [166, 86, 40]/255, [255, 255, 51]/255, [255, 127, 0]/255, ...
          [152, 78, 163]/255, [77, 175, 74]/255}; 


%%
% regularization
epsilon = (.08)^2;
%epsilon = 1e-4;

% data setting
mu = mu([1,2]);
col = col([1,2], :);
m = 5; % number of barycenter (including end pts)
savefig = 1;


% plot input 
for i = 1 : length(mu)
    opt.color = col(i,:);
    clf;
    dir = [rep 'input-' num2str(i) '.pdf'];
    mydisplayfn(reshape(mu{i}, [2 2 n n]),n,opt,dir,savefig);
end

%% 
% linear interp
opt.nb_ellipses = n;
for k1=1:m    
    t1 = (k1-1)/(m-1);
    w = [(1-t1), t1];
    
    if ~ismember(1, w)
        opt.color = sum( col .* repmat(w', [1 3]) );
        clf;
        nu0{k1} = 0;
        for ii = 1:length(w)
            nu0{k1} = nu0{k1} + w(ii) * mu{ii};
        end
        dir = [rep 'linearinterp-' num2str(k1) '-' num2str(m) '.pdf'];
        mydisplayfn(reshape(nu0{k1}, [2 2 n n]),n,opt,dir,savefig);
    end
    
end


%% cvx as baseline
lambda = epsilon;
for k1=1:m
    t1 = (k1-1)/(m-1);
    w = [(1-t1), t1];
    if ~ismember(1, w)
        cvx_begin 
            variable Gamma1(d,d,N,N) semidefinite;
            variable Gamma2(d,d,N,N) semidefinite;
            variable Xbar(d,d,N) semidefinite;
            obj = 0;
            for mm = 1:N
                for nn = 1:N
                    Gamma1ij = Gamma1(:,:,mm,nn);
                    Gamma2ij = Gamma2(:,:,mm,nn);
                    Cij = c(:,:,mm,nn);
                    if lambda < 1e-10
                        obj = obj + w(1) * trace(Cij * Gamma1ij ) + w(2) * trace(Cij * Gamma2ij );
                    else
                        obj = obj + w(1) * (trace(Cij * Gamma1ij ) - lambda * quantum_entr(Gamma1ij) - lambda*trace(Gamma1ij)) + ...
                                    w(2) * (trace(Cij * Gamma2ij ) - lambda * quantum_entr(Gamma2ij) - lambda*trace(Gamma2ij));
                    end
                end
            end
            minimize( obj )
            subject to
                reshape(sum(Gamma1, 4), [d d N]) - Xbar == 0;
                reshape(sum(Gamma1, 3), [d d N]) - mu{1} == 0;
                reshape(sum(Gamma2, 4), [d d N]) - Xbar == 0;
                reshape(sum(Gamma2, 3), [d d N]) - mu{2} == 0;
                reshape(sum(Xbar, 3), [d d]) == eye(d);
        cvx_end 
    
        obj_cvx(k1) = cvx_optval;
    end
end


%%


options.lambda = epsilon;
options.MOTiter = 100; % inner iteration for each coupling
options.WBiter = 10; % outer iteration for WB
options.tolblockDS = 1e-6;
options.WBmethod = 'SD';

inits = {'li', 'qot', 'uf'};
inits = {'qot'};

infos = cell(m,1);
for k1=1:m
    t1 = (k1-1)/(m-1);
    w = [(1-t1), t1];
    
    if ~ismember(1, w)
    
        info_inits = cell(length(inits), 1);
        for i_init = 1 : length(inits)
            init = inits{i_init};

            info_inits{i_init}.init_type = init;

            % initialize
            switch init
                case 'li'
                    % linear interpolation
                    nuinit = 0;
                    for ii = 1 : length(w)
                        nuinit = nuinit + w(ii) * mu{ii};
                    end

                case 'qot'
                    % initalize from qot 
                    rho = 1;
                    qotopts.niter = 100;
                    qotopts.disp_rate = NaN;
                    qotopts.over_iterations = 10; % seems important to avoid oscilations
                    fprintf('QOT init ...\n');
                    nuinit= quantum_barycenters(mu,c,rho,epsilon,w,qotopts);
                    nuinit = spdsimplexnormalize(nuinit);

                case 'mu'
                    % use the first marginal as init
                    nuinit = mu{1};

                case 'uf'
                    % uniform
                    temp = randn(d);
                    for ii = 1 : (N)
                        nuinit(:,:,ii) = eye(d);
                    end
                    nuinit = spdsimplexnormalize(nuinit);

                case 'rand'
                    % random init
                    nuinit = [];
            end
        
        
            fprintf('Barycenter (%d/%d):\n', k1, m);
            [nu, gamma, info_mot] = BlockWBarycenter(mu,c,w,options, nuinit);
            fprintf('\n');
        
            info_inits{i_init}.xbar = nu;
            info_inits{i_init}.gamma = gamma;
            info_inits{i_init}.info_mot = info_mot;
            
            % plot barycenter
            opt.color = sum( col .* repmat(w', [1 3]) );
            dir = [rep 'MOTbarycenter-' init '-' num2str(k1) '-' num2str(m) '.pdf'];
            mydisplayfn(reshape(nu, [2 2 n n]),n,opt,dir,savefig);
                
        end
        
        infos{k1} = info_inits;
            
    end
end

%% plot convergence against cvx_optval


for k1 = 1:m
    t1 = (k1-1)/(m-1);
    w = [(1-t1), t1];
    
    if ~ismember(1, w)
        h = figure();
        plot(221);

        info_inits = infos{k1};
        for i_init = 1 : length(inits)

            cvxobj = obj_cvx(k1);

            info = info_inits{i_init}.info_mot;
            semilogy([info.iter], abs([info.cost] - cvxobj), '-o', 'color', colors{i_init}, 'LineWidth', lw, 'MarkerSize',ms);  hold on;
            Legend{i_init}= [info_inits{i_init}.init_type '-init-' num2str(k1)];

        end
        ax1 = gca;
        set(ax1,'FontSize', fs);
        set(h,'Position',[100 100 600 500]);
        xlabel('Iteration', 'fontsize', fs);
        ylabel('Optimality gap', 'fontsize', fs);
        lg = legend(Legend);
    end
end
%% QOT
rho = 1;
options1.disp_rate = NaN;
options1.over_iterations = 10; % seems important to avoid oscilations
options1.niter = 100; % sinkhorn #iterates
%options1.nu_init = nuinit;

% computations
for k1=1:m
    t1 = (k1-1)/(m-1);
    w = [(1-t1), t1];
    if ~ismember(1, w)
        fprintf('Barycenter (%d/%d):\n', k1, m);
        [nu2{k1},gamma,err] = quantum_barycenters(mu,c,rho,epsilon,w, options1);
        nu2{k1} = spdsimplexnormalize(nu2{k1});
    end
end

% display
nu2{1} = mu{1};
nu2{m} = mu{2};
for k1=1:m
    t1 = (k1-1)/(m-1);
    w = [(1-t1), t1];
    if ~ismember(1, w)
        opt.color = sum( col .* repmat(w', [1 3]) );
        dir = [rep 'QOTbarycenter-' num2str(k1) '-' num2str(m) '.pdf'];
        mydisplayfn(nu2{k1},n,opt,dir,savefig);
    end
end





%% display helper fn
function mydisplayfn(mu,n,opt,dir,savefig)
    % opt is for plot_tesnros_2d
    opt.nb_ellipses = n;
    clf;
    plot_tensors_2d(reshape(mu, [2 2 n n]), opt);
    drawnow;
    if savefig == 1
        saveas(gcf, dir, 'png');
    end
end
