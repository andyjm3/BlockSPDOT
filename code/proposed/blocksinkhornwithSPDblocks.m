function [B, Rtilde, Ctilde, gaps, times] = blocksinkhornwithSPDblocks(A0, M, N, maxiter, checkperiod, tol, verbosity)
% blocksinkhornwithSPDblocks implements sinkhorn type iterations at blocks
% by iteratively normalizing row blocks and column blocks. Each block is SPD.
% A is d-by-d-by-m-by-n, i.e., Aij is a SPD matrix.
% M is d-by-d-by-m, i.e., Mi is a SPD matrix.
% N is d-by-d-by-n, i.e., Nj is a SPD matrix.
    
  d = size(A0, 1);
  m = size(A0, 3);
  n = size(A0, 4);

  myI = eye(d);
  

  if ~exist('M', 'var') || isempty(M)
      for i = 1:m
          M(:,:,i) = myI/m;
      end
  end

  if ~exist('N', 'var') || isempty(N)
      for j = 1:n
          N(:,:,j) = myI/n;
      end
  end

  if ~exist('maxiter', 'var') || isempty(maxiter)
      maxiter = d*d*n*m;
  end
  
  if ~exist('checkperiod', 'var') || isempty(checkperiod)
      checkperiod = 100;
  end

  if ~exist('tol', 'var') || isempty(tol)
      tol = eps;
  end

  if ~exist('verbosity', 'var') || isempty(verbosity)
      verbosity = 0;
  end

  A = A0;

  iter = 0;
  
  C = computeC(A, N);
  A = computeCACt(A, C);
  R = computeR(A, M);
  A = computeRARt(A, R);

  % Rtilde and Ctilde: R and C terms
  Ctilde = C; 
  Rtilde = R;
  
  Ctildes = Ctilde;
  Rtildes = Rtilde;
  
  gaps = [];
  times = [];
  gap = inf;
  timetic = tic();
  while iter < maxiter
    iter = iter + 1;
    if mod(iter, checkperiod) == 0 
        gap = computegap(A, M, N);
        % store gap and time
        gaps = [gaps gap];
        if isempty(times)
            times = [times toc(timetic)];
            timetic = tic();
        else
            timeelapse = times(end) + toc(timetic);
            times = [times timeelapse];
            timetic = tic();
        end
        
        if isnan(gap)
            break;
        end
        if gap <= tol
            break;
        end
    end
    C = computeC(A, N);
    A = computeCACt(A, C);
    R = computeR(A, M);
    A = computeRARt(A, R);

    % Update Rtilde and Ctilde
    Rtilde = blockmult(Rtilde, R);
    Ctilde = blockmult(Ctilde, C);
    
    %Ctildes = cat(4, Ctildes, Ctilde);
    %Rtildes = cat(4, Rtildes, Rtilde);
    
  end
  B = A;
  if verbosity
    fprintf('BlockSinkIter %4d\t gap %e \n', iter, gap);
  end

  Rtilde = blocksymm(Rtilde);
  Ctilde = blocksymm(Ctilde);
  
  

  % % The following B1 and B2 are very close to B,
  % % implying that Rtilde and Ctilde are commutative.
  % % Also, "Bij = Rtilde_i Ctilde_j A0_ij  Ctilde_j Rtilde_i".
  % % Ris and Cjs are all symmetric asymptotically.

  %B1 = computeRARt(computeCACt(A0, Ctilde), Rtilde);
  %B2 = computeCACt(computeRARt(A0, Rtilde), Ctilde);
  %norm(B1(:,:,1,1) - B(:,:,1,1),'fro')/norm(B(:,:,1,1),'fro')
  %norm(B2(:,:,1,1) - B(:,:,1,1),'fro')/norm(B(:,:,1,1),'fro')
  %keyboard; 
end


function Deltasymm = symm(Delta)
  Deltasymm = 0.5*(Delta + Delta');
end 


function symmDblock = blocksymm(Dblock)
  l = size(Dblock, 3);
  symmDblock = Dblock;
  for ll = 1 : l
    symmDblock(:,:,ll) = symm(Dblock(:,:,ll));
  end 
end


function LLtilde = blockmult(Ltilde, L) % L times Ltilde
  l = size(L, 3);
  LLtilde = nan(size(Ltilde));
  for i = 1 : l
    LLtilde(:,:,i) = L(:,:,i)*Ltilde(:,:,i); % Left multipication
  end
end


function RDRt = computeRARt(D, R)
  % L can be either R or C
  m = size(D, 3);
  n = size(D, 4);
  RDRt = nan(size(D));  
  for i = 1 : m
    Rtemp = R(:,:,i);
    for j = 1 : n
    RDRt(:,:,i,j) = Rtemp * D(:,:,i,j) * Rtemp';
    end
  end


end

function CDCt = computeCACt(D, C)
  % L can be either R or C
  m = size(D, 3);
  n = size(D, 4);
  CDCt = nan(size(D));
  for j = 1 : n
    Ctemp = C(:,:,j);
    for i = 1 : m
    CDCt(:,:,i,j) = Ctemp * D(:,:,i,j) * Ctemp';
    end
  end
end


function R = computeR(D, M)
  d = size(D, 1);
  m = size(D, 3);
  Dsum = reshape(sum(D, 4), [d d m]);
  R = nan(size(Dsum));
  for i = 1 : m
    R(:,:,i) = computericcati(Dsum(:,:,i), M(:,:,i));
  end
end


function C = computeC(D, N)
  d = size(D, 1);
  n = size(D, 4);
  Dsum = reshape(sum(D, 3), [d d n]);
  C = nan(size(Dsum));
  for j = 1 : n
    C(:,:,j) = computericcati(Dsum(:,:,j), N(:,:,j));
  end
end


function X = computericcati(P, Q)
% Solve for X in XPX = Q
  % % Costly
  % Phalf = sqrtm(P);
  % X = symm((Phalf \ sqrtm(Phalf*Q*Phalf))  / Phalf);

  X = symm(real(P \ sqrtm(P*Q)));

  % % debug
  % norm(X*P*X - Q, 'fro')
  % keyboard
end


function [gap, gapM, gapN] = computegap(D, M, N)
  d = size(D, 1);
  m = size(M, 3);
  n = size(N, 3);
  errorM = nan(1, m);
  errorN = nan(1, n);
  sumDrow = reshape(sum(D, 4), [d d m]);
  sumDcol = reshape(sum(D, 3), [d d n]);
  for i = 1 : m
    errorM(1, i) = norm(sumDrow(:,:,i) - M(:,:,i),'fro')/norm(M(:,:,i),'fro');
  end
  for j = 1 : n
    errorN(1, j) = norm(sumDcol(:,:,j) - N(:,:,j),'fro')/norm(N(:,:,j),'fro');
  end
  gapM = max(errorM);
  gapN = max(errorN);
  gap = gapM + gapN;
end

