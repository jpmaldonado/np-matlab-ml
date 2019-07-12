clear;

data = csvread('../data/demo/nlsdata.csv');

X = data(:, 1:2);
y = data(:, 3);

rng(42) % Random number generator seed

svm = fitcsvm(X,y, 'KernelFunction', 'gaussian');
cv = crossval(svm, 'KFold', 7); % Cross validation like in Graphic tool
loss = kfoldLoss(cv); % Average cv error

