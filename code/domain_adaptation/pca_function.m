function [data_train_pcaed,data_test_pcaed,Vk] = pca_function(data_train,data_test)
% data_train should be numTrain x numFeat matrix
% data_test should be numTest x numFeat matrix
% code assumed data_train and data_test given

explained_variance = 1;
fprintf('explained variance: %g\n',explained_variance);
numTrain = size(data_train,1);
% numTest = size(data_test,1);
numFeat = size(data_train,2);

trainMean = mean(data_train);
train_tilde = data_train-trainMean;
size(train_tilde)
M = (train_tilde'*train_tilde)/(numTrain-1);
[evec, eval] = eig(M);
total = sum(sum(eval));
eval = max(eval);

for i = 0:(numFeat-1)
    test = sum(eval((numFeat-i):numFeat))/total;
    if test >= explained_variance
        k = i;
        fprintf('chosen k is %d\n',k+1);
        break
    end
end
Vk = evec(:,((numFeat-k):numFeat));%last has the highest eigenvalue

data_train_pcaed= train_tilde*Vk; 
if isempty(data_test)
    data_test_pcaed = [];
else
    test_tilde = data_test-trainMean;
    data_test_pcaed = test_tilde*Vk;
end
