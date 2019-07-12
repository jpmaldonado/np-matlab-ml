clear;
fname = '../../data/exercises/FTIRSpectraInstantCoffee/FTIR_Spectra_instant_coffee.csv';
X = csvread(fname, 3, 1, [3 1 288 56]);
y = csvread(fname, 1, 1, [1 1 1 56]);
X = X';
y = y';

[Xtrain, Xval, ytrain, yval] = TrainValSplit(X,y, 0.9);


svm = fitcsvm(Xtrain, ytrain);
cv = crossval(svm);
loss = kfoldLoss(cv, 'mode', 'individual');
hist(loss)

% almost duplicate? size(uniquetol(X,0.07,'ByRows',true))
confusionmat(yval, predict(svm, Xval))

% PCA evaluation
% corr(X) %  lots of correlated features!
Xpca = pca(X)

figure 
hold on
scatter(Xpca(y==1,1), Xpca(y==1,2), '.b')
scatter(Xpca(y==2,1), Xpca(y==2,2), '.r')

