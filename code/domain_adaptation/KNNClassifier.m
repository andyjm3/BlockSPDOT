function [Result] = KNNClassifier(Dmap, Y_train, Y_test, k)
%KNNCLASSIFIER_MN Calculates the prediction of knn and its accuracy for
%MNIST dataset
%   Input: Dmap is N_train-by-N_test
%        : k is number of neighbours; dist represents the distance metric
%   Output: [Accuracy, prediction y of test set]

Ntest = size(Dmap, 2);
pred_y = zeros(1,Ntest);

for i=1:Ntest
    try
        [~,idx] = mink(Dmap(:,i), k);
    catch
        idx = min_k(Dmap(:,i), k);
    end
    pred_y(i) = mode(Y_train(idx));
end

%[~,Result,~]= confusion.getMatrix(Y_test,pred_y,0);
Result = sum(pred_y==Y_test)/length(Y_test);
end

