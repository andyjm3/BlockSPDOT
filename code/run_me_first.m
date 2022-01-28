% Add folders to path.

addpath(pwd);

cd manopt;
addpath(genpath(pwd));
cd ..;

cd auxiliary
addpath(genpath(pwd));
cd ..;

cd proposed;
addpath(genpath(pwd));
cd ..;

cd quantumot/cvx;
addpath(genpath(pwd));
cd ../..;

cd quantumot/qot;
addpath(genpath(pwd));
cd ../..;

cd quantumot/cvxquad;
addpath(genpath(pwd));
cd ../..;

cd GW;
addpath(genpath(pwd));
cd ..;

cd quantumot;
addpath(genpath(pwd));
cd ..;

cd domain_adaptation;
addpath(genpath(pwd));
cd ..;
