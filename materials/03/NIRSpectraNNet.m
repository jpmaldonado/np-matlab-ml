clear; clc;

data = readtable('../../data/exercises/nir/cal2018.csv');

data = table2array(data);
X = data(:, 3:end);
y = data(:,2);

[Xtrain, Xval, ytrain, yval] = TrainValSplit(X, y, 0.8);

net = feedforwardnet([8 3]);
% view(net)
trainedNet = train(net, Xtrain', ytrain');

ypred = trainedNet(Xval');