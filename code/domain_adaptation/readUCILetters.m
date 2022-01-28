function [train, labels_train, test, labels_test] = readUCILetters()
    % mainPath = strcat('.',filesep,'UCI_Letters',filesep);
    path = strcat('letter-recognition.data');
    A = importdata(path);
    data = (A.data'); 
    labels = cell2mat(A.textdata);
    train = data(:,1:16000);
    labels_train = labels(1:16000);
    test = data(:,16001:end);
    labels_test = labels(16001:end);
end