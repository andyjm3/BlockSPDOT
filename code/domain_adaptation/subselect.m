function [datasub, labelsub] = subselect(data, labels, Nsub, prob_vector)
% random subset selection based on class, data is a cell array of length N

class = unique(labels);
n_class = length(class);

if ~exist('prob_vector', 'var') || isempty(prob_vector)
    prob_vector = (1/n_class)*ones(1, n_class);
end

sample_per_class_list = floor(Nsub.*prob_vector);
remaining_samples = Nsub-sum(sample_per_class_list);
if remaining_samples>0
    class_permute = randperm(n_class);
    chosen_class = class_permute(1:remaining_samples);
    sample_per_class_list(chosen_class) = sample_per_class_list(chosen_class)+1;
end

%sample_per_class_list = round(Nsub.*prob_vector);
% keyboard;

datasub = {};
labelsub = [];
for i = 1:n_class
    sample_per_class = sample_per_class_list(i);
    if sample_per_class>0
        classi = class(i);
        classi_idx = (labels == classi);
        data_i = data(classi_idx);
        N_permute = randperm(length(data_i));
    %     if i < n_class
        datasub = [datasub data_i(N_permute(1:sample_per_class))];
        labelsub = [labelsub classi * ones(1,sample_per_class)];
    %     else
    %         Ni = Nsub - length(datasub);
    %         datasub = [datasub data_i(N_permute(1:Ni))];
    %         labelsub = [labelsub classi * ones(1,Ni)];
    %     end
    end
end
% keyboard
% final shuffle
N_permute = randperm(Nsub);

datasub = datasub(N_permute);
labelsub = labelsub(N_permute);

end