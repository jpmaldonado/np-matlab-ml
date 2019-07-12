data = csvread('../data/demo/nlsdata.csv');

X = data(:, 1:2);
y = data(:, 3);

rng(42) % Random number generator seed

svm = fitcsvm(X,y);
cv = crossval(svm, 'KFold', 7); % Cross validation like in Graphic tool
loss = kfoldLoss(cv); % Average cv error
losses = kfoldLoss(cv, 'Mode', 'individual');
bar(losses);

% Get the folds explicitly

cvp = cvpartition(y, 'KFold',7);

for i = 1:cvp.NumTestSets
    Xtrain = X(cvp.training(i), :); % Indices for training in 1st fold
    ytrain = y(cvp.training(i));
    Xtest = X(cvp.test(i), :); % Indices for evaluation
    ytest = y(cvp.test(i));
    svm = fitcsvm(Xtrain, ytrain); % Train model
    ypreds = predict(svm, Xtest);
    loss = sum(ypreds~=ytest)/numel(ytest);
    if(loss>0.1)
         disp(loss)
    end
end
