clear
clc
rng('default');
rand_seeds = [1,11,21,31,41];

% subsample
filename = 'letter_pca100'; %fashionmnist_pca75,mnist_pca75,letter_pca100
m = 250;
n = 50;
d = 5;
k = d;
frac_vec = [0.1, 0.2, 0.3, 0.4, 0.5];
skewed_class = 1; %integer in the range [1,10]


scalarOT_cvx_result = nan(length(frac_vec),length(rand_seeds)); %for ScalarOT cvx results
matrixOT_cvx_result = nan(length(frac_vec),length(rand_seeds)); %for proposedOT cvx results
projmatrixOT_result = nan(length(frac_vec),length(rand_seeds)); %for projected proposedOT results
spdOT_cvx_result = nan(length(frac_vec),length(rand_seeds)); %for spdOT cvx results
qOT_cvx_result = nan(length(frac_vec),length(rand_seeds)); %for qOT cvx results
% Given X_src as d by d by m and y_src as vector of size 1 by m
% Given X_tgt as d by d by n and y_tgt as vector of size 1 by n
% Each object is a SPD matrix.

load(sprintf('%s.mat',filename));

fprintf('file %s loaded\n',filename);

numFeat = size(data_train,1);
n_class = length(unique(labels_train));

data_train = data_train/15; %division by 255 for fashionMNIST, 1 for mnist, 15 for letter
data_test = data_test/15; %division by 255 for fashionMNIST, 1 for mnist, 15 for letter

[X_train_all, y_train_all] = creatgroups(data_train, labels_train', k);
[X_test_all, y_test_all] = creatgroups(data_test, labels_test', k);

for randiter = 1:length(rand_seeds)
    rng(rand_seeds(randiter));
    for fraciter = 1:length(frac_vec)
        frac = frac_vec(fraciter);
        probvectortrain = ones(1,n_class);
        probvectortrain(skewed_class) = (n_class-1)*frac/(1-frac);
        probvectortrain = probvectortrain/sum(probvectortrain);
        probvectortest = ones(1,n_class);
        probvectortest = probvectortest/sum(probvectortest);
        
        [X_train, y_train] = subselect(X_train_all, y_train_all, m, probvectortrain);
        [X_test, y_test] = subselect(X_test_all, y_test_all, n, probvectortest);
        
        
        %% X_src and X_tgt are the covariances
        X_src = nan(d,d,length(X_train));
        for ii = 1 : length(X_train)
            X_src(:,:,ii) = ( (X_train{ii}(numFeat-(d-1):end,:))*(X_train{ii}(numFeat-(d-1):end,:))' ) ./k;
        end
        
        X_tgt = nan(d,d,length(X_test));
        for jj = 1 : length(X_test)
            X_tgt(:,:,jj) = ( (X_test{jj}(numFeat-(d-1):end,:))*(X_test{jj}(numFeat-(d-1):end,:))' ) ./k;
        end
        
%         ridge = 1e-8*eye(d);
%         X_src_r = nan(d,d,length(X_train));
%         for ii = 1 : length(X_train)
%             X1 = ( (X_train{ii}(numFeat-(d-1):end,:))*(X_train{ii}(numFeat-(d-1):end,:))' ) ./k;
%             X_src_r(:,:,ii) = (X1 + X1')/2 + ridge;
%         end
%         X_tgt_r = nan(d,d,length(X_test));
%         for jj = 1 : length(X_test)
%             X2 = ( (X_test{jj}(numFeat-(d-1):end,:))*(X_test{jj}(numFeat-(d-1):end,:))' ) ./k;
%             X_tgt_r(:,:,jj) = (X2 + X2')/2 + ridge;
%         end
        
        y_src = y_train;
        y_tgt = y_test;
        
        % compute distance matrices
%         Cmw = zeros(d,d,m,n);
        Cq = zeros(d,d,m,n);
        Cw = zeros(m,n);
%         Cr = zeros(m,n);
        for ii = 1:m
            for jj = 1:n
                Cw(ii,jj) = norm(X_src(:,:,ii) - X_tgt(:,:,jj), 'fro')^2 ;
%                 Cmw(:,:,ii,jj) = (X_src(:,:,ii) - X_tgt(:,:,jj)) * (X_src(:,:,ii) - X_tgt(:,:,jj))';
%                 Cr(ii,jj)  = RiemannianDist(X_src_r(:,:,ii),  X_tgt_r(:,:,jj), 2)^2; % rdist square
                Cq(:,:,ii,jj) = norm(X_src(:,:,ii) - X_tgt(:,:,jj), 'fro')^2 * eye(d);
            end
        end
        
        
        %% Run algorithms
        % set marginals as uniform distribution
        p = ones(m,1)./m;
        q = ones(n,1)./n;
        
        P = repmat(eye(d)./m, [1 1 m]);
        Q = repmat(eye(d)./n, [1 1 n]);
        
        
        %% W
        cvx_begin
            variable H(m,n)
            minimize sum(sum(H.*Cw))
            subject to
            H*ones(n,1)==p;
            H'*ones(m,1)==q;
            H>=0;
        cvx_end
        gamma_w = H;
        
        X_src_w = computeprojectionW(X_tgt, gamma_w);
        Dist_w = computeDist(X_src_w, X_tgt);
        result_w = KNNClassifier(Dist_w, y_src, y_tgt, 1);
        fprintf('randiter: %d, fraciter: %d, CVX W22 accuracy: %f\n',randiter,fraciter,result_w);
        scalarOT_cvx_result(fraciter,randiter) = result_w;
        disp(scalarOT_cvx_result);% for bookeeping
        
        %% SPT-OT
%         cvx_begin
%             variable G(m,n)
%             minimize sum(sum(G.*Cr))
%             subject to
%             G*ones(n,1)==p;
%             G'*ones(m,1)==q;
%             G>=0;
%         cvx_end
%         gamma_r = G;
%         
%         X_src_r = computeprojectionWriemannian(X_tgt_r, gamma_r);
%         Dist_r = computeDistriemannian(X_src_r, X_tgt_r);
%         result_r = KNNClassifier(Dist_r, y_src, y_tgt, 1);
%         fprintf('randiter: %d, fraciter: %d, CVX SPD-OT accuracy: %f\n',randiter,fraciter,result_r);
%         spdOT_cvx_result(fraciter,randiter) = result_r;
%         disp(spdOT_cvx_result);% for bookeeping
        
        %% Quantum OT with KL reg
        rho1 = 10; % regularization parameter
        rho2 = 10;

        cvx_begin
             variable Gamma(d,d,m,n) semidefinite;
             obj = 0;
             for ii = 1 : m
                 for jj = 1 : n
                     obj = obj + trace((Gamma(:,:,ii,jj)) * Cq(:,:,ii,jj));
                 end
             end

             sumGamma3 = reshape(sum(Gamma, 3), [d d n]);
             sumGamma4 = reshape(sum(Gamma, 4), [d d m]);

             for ii = 1 : m
                 obj = obj + rho1 * quantum_rel_entr(sumGamma4(:,:,ii), P(:,:,ii));
             end

             for jj = 1 : n
                 obj = obj + rho2 * quantum_rel_entr(sumGamma3(:,:,jj), Q(:,:,jj));
             end

             minimize( obj )
        cvx_end
        
        Gamma_q = Gamma;
        X_src_q = computeprojectionMWlyap(X_tgt, Gamma_q); % symmetric
        Dist_q = computeDist(X_src_q, X_tgt);
        result_q = KNNClassifier(Dist_q, y_src, y_tgt, 1);
        fprintf('randiter: %d, fraciter: %d, CVX QOT accuracy: %f\n',randiter,fraciter,result_q);
        qOT_cvx_result(fraciter,randiter) = result_q;
        disp(qOT_cvx_result);% for bookeeping
        
        %% Matrix W
%         cvx_begin
%         variable Gamma(d,d,m,n) semidefinite;
%         obj = 0;
%         for ii = 1 : m
%             for jj = 1 : n
%                 obj = obj + trace((Gamma(:,:,ii,jj)) * Cmw(:,:,ii,jj));
%             end
%         end
%         minimize( obj )
%         subject to
%         reshape(sum(Gamma, 4), [d d m]) - P == 0;
%         reshape(sum(Gamma, 3), [d d n]) - Q == 0;
%         cvx_end
%         Gamma_mw_cvx = Gamma;
%         
%         X_src_mw_cvx = computeprojectionMWlyap(X_tgt, Gamma_mw_cvx); % symmetric
%         Dist_mw_cvx = computeDist(X_src_mw_cvx, X_tgt);
%         result_mw_cvx = KNNClassifier(Dist_mw_cvx, y_src, y_tgt, 1);
%         fprintf('randiter: %d, fraciter: %d, CVX MOT accuracy: %f\n',randiter,fraciter,result_mw_cvx);
%         matrixOT_cvx_result(fraciter,randiter) = result_mw_cvx;
%         disp(matrixOT_cvx_result);% for bookeeping
        
        %% MOT
        %         options_mot.maxiter = 20;
        %         options_mot.tolgradnorm = 1e-12;
        %         Gamma_mw_mot = BlockMOT(Cmw, options_mot);
        %         X_src_mw_mot = computeprojectionMWlyap(X_tgt, Gamma_mw_mot); % symmetric
        %         Dist_mw_mot = computeDist(X_src_mw_mot, X_tgt);
        %         result_mw_mot = KNNClassifier(Dist_mw_mot, y_src, y_tgt, 1;
        
        
        %% ProjMOT
        % X_src and X_tgt are the covariances
%         X_src_projmot = nan(numFeat,numFeat,length(X_train));
%         for ii = 1 : length(X_train)
%             X_src_projmot(:,:,ii) = ( X_train{ii}*X_train{ii}' ) ./k;
%         end
%         
%         X_tgt_projmot = nan(numFeat,numFeat,length(X_test));
%         for jj = 1 : length(X_test)
%             X_tgt_projmot(:,:,jj) = ( X_test{jj}*X_test{jj}' ) ./k;
%         end
%         
%         y_src = y_train;
%         y_tgt = y_test;
%         
%         % compute distance matrices
%         Cmw_projmot = zeros(numFeat,numFeat,m,n);
%         for ii = 1:m
%             for jj = 1:n
%                 Cmw_projmot(:,:,ii,jj) = (X_src_projmot(:,:,ii) - X_tgt_projmot(:,:,jj)) * (X_src_projmot(:,:,ii) - X_tgt_projmot(:,:,jj))';
%             end
%         end
        %         r = 5;
        %         options_projmot.maxiter = 500;
        %         options_projmot.tolgradnorm = 1e-12;
        %         [WGammaProjMOT] = ProjBlockMOT(Cmw_projmot, r, options_projmot);
        %         [X_src_mw_projmot, X_tgt_mw_projmot] = computeprojectionprojMW(X_tgt_projmot, WGammaProjMOT.W, WGammaProjMOT.Gamma);
        %         Dist_mw_projmot = computeDist(X_src_mw_projmot, X_tgt_mw_projmot);
        %         result_mw_projmot = KNNClassifier(Dist_mw_projmot, y_src, y_tgt, 1);
        %         projmatrixOT_result(frac_iter,randiter) = result_mw_projmot;
        %         fprintf('randiter: %d, fraciter: %d, projMOT accuracy: %f\n',randiter,fraciter,result_mw_projmot);
        %         disp(projmatrixOT_result);% for bookeeping
        
    end
end

scalarOT_cvx_mean = mean(scalarOT_cvx_result,2);
matrixOT_cvx_mean = mean(matrixOT_cvx_result,2);
projmatrixOT_mean = mean(projmatrixOT_result,2);
spdOT_cvx_mean = mean(spdOT_cvx_result,2);
qOT_cvx_mean = mean(qOT_cvx_result,2);
scalarOT_cvx_std = std(scalarOT_cvx_result,[],2);
matrixOT_cvx_std = std(matrixOT_cvx_result,[],2);
projmatrixOT_std = std(projmatrixOT_result,[],2);
spdOT_cvx_std = std(spdOT_cvx_result,[],2);
qOT_cvx_std = std(qOT_cvx_result,[],2);
fprintf('fractions\n');
disp(frac_vec);
fprintf('scalarOT mean\n');
disp(scalarOT_cvx_mean');
fprintf('scalarOT std\n');
disp(scalarOT_cvx_std');
fprintf('matrixOT cvx mean\n');
disp(matrixOT_cvx_mean');
fprintf('matrixOT cvx std\n');
disp(matrixOT_cvx_std');
fprintf('spdOT cvx mean\n');
disp(spdOT_cvx_mean');
fprintf('spdOT cvx std\n');
disp(spdOT_cvx_std');
fprintf('qOT cvx mean\n');
disp(qOT_cvx_mean');
fprintf('qOT cvx std\n');
disp(qOT_cvx_std');



