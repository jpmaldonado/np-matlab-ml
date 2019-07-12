clear; clc;
data = readtable('../../data/exercises/cheddar-cheese.csv');

data = table2array(data);

X = data(:, 2:3);
y = data(:,5);

[Xtrain, Xval, ytrain, yval] = TrainValSplit(X,y, 0.8);

gp = fitrgp(Xtrain, ytrain, 'OptimizeHyperparameters', 'auto');
cv = crossval(gp, 'Holdout', 0.1);
loss = kfoldLoss(cv);

% Inference
[ypred, ysd, yint] = predict(gp, Xtrain);

% Diagnostic: Prediction vs true
figure 
hold on
plot(Xtrain(:,1), ytrain, '.r')
plot(Xtrain(:,1), ypred, '+b')
legend('True', 'Predicted')
xlabel('Acetic')
ylabel('Taste')
errorbar(Xtrain(:,1), ypred, ysd, '.c')
hold off


% Which points have higher uncertainty?
x1min = min(Xtrain(:,1));
x1max = max(Xtrain(:,1));

x2min = min(Xtrain(:,2));
x2max = max(Xtrain(:,2));

x1Mesh = linspace(x1min, x1max, 100);
x2Mesh = linspace(x2min, x2max, 100);

[X1, X2] = meshgrid(x1Mesh, x2Mesh);
fun = @(z) predict(gp, z);

Z = zeros(100, 100);
SD = zeros(100, 100);

for i=1:100
    for j=1:100
        [Z(i,j), SD(i,j)] = fun([X1(i,j), X2(i,j)]);
    end
end

surf(X1, X2, Z, SD)

