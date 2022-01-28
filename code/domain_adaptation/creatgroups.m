function [X, labels] = creatgroups(data, labels_data, k)
% labels_data is row vector.
% X is cell
% lables is row.


classes_data = unique(labels_data);


X = {};
labels = nan(1, 1e5);

m = 0;
for ii = 1: length(classes_data)
    myidx = find(labels_data == classes_data(ii));
    
    myX = data(:,myidx);
    [~, myXpart]=randDivide(myX', floor(length(myidx)/k));
    
    for jj = 1:length(myXpart)
        m = m + 1;
        % we may see more than k samples in myXpart{jj}, so pruning.
        temp = myXpart{jj};;
        X{m} = temp(:,1:k); 

        labels(m) = ii; %classes_data(ii); % this assumes that we see all the classes
    end
    % ll = size(X{m},2);
    % if ll ~= k
    %     keyboard;
    % end
end
labels(m+1:end) = [];
end


function [idxo prtAt]=randDivide(M,K)
[n,m]=size(M);
np=(n-rem(n,K))/K;
B=M;
[c,idx]=sort(rand(n,1));
C=M(idx,:);
i=1;
j=1;
ptrA={};
idxo={};
n-mod(n,K);
while i<n-mod(n,K)
    prtA{j}=C(i:i+np-1,:);
    idxo{i}=idx(i:i+np-1,1);
    i=i+np;
    j=j+1;
end

% prtA{j}=C(i:n,:);

prtAt={};
% transpose it
for jj = 1:length(prtA)
    prtAt{jj} = prtA{jj}';
end

end