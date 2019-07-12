clear;
clc; % clear command window

data = csvread('../data/demo/lsdata.csv');
X = data(:,1:2); 
y = data(:,3);

y = y>0; % logical condition 0/1 

[X_train, X_val, y_train, y_val] = TrainValSplit(X, y, 0.8);

% Goal: Classification model for our data

logreg = fitglm(X_train, y_train, ...
    'Distribution', 'binomial', 'Link', 'logit');

% Model validation
y_prob = predict(logreg, X_val);

% Get classes from proba
y_pred = y_prob > 0.5;

% Model performance: confusion matrix
confusionmat(y_val, y_pred);


% How does the data look?
figure 
hold on % many things on same canvas
scatter(X(y==0,1),X(y==0,2),'.b')
scatter(X(y==1,1),X(y==1,2),'.r')
title('My data')
legend('Zero class', 'Positive class')
hold off









