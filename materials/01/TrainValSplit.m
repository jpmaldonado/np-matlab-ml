function [X_train, X_val, y_train, y_val] = TrainValSplit(X, y, trainFrac)
% Input: Independent and dependent variables, fraction of train set
% Output: Data partition on train and validation

[n_samples, ~] = size(X);

idxs = 1:n_samples;
rng(1); % Random number generator seed

trainIdxs = randsample(n_samples, round(trainFrac*n_samples));
valIdxs = idxs(~ismember(idxs, trainIdxs))'; 

X_train = X(trainIdxs, :);
y_train = y(trainIdxs);
X_val = X(valIdxs,:);
y_val = y(valIdxs);

end