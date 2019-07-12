clear; clc;

data = readtable('../../data/exercises/nir/cal2018.csv');

data = table2array(data);
X = data(:, 3:end);
y = data(:,2);

[Xtrain, Xval, ytrain, yval] = TrainValSplit(X, y, 0.8);

net = newrbe(Xtrain', ytrain');

ypred = sim(net, Xval'); % fitting + validation