data = csvread('../data/demo/nlsdata.csv');

X = data(:, 1:2);
y = data(:, 3);

svm = fitcsvm(X,y, 'KernelFunction', 'gaussian');

% Plot decision boundary
x1min = min(X(:,1));
x2min = min(X(:,2));
x1max = max(X(:,1));
x2max = max(X(:,2));

x1Mesh = linspace(x1min, x1max, 100);
x2Mesh = linspace(x2min, x2max, 100);

[X1, X2] = meshgrid(x1Mesh, x2Mesh);
fun = @(z) double(predict(svm, z));
Z = fun([X1(:), X2(:)]);

figure
hold on
scatter(X(y==1, 1), X(y==1, 2), '*b')
scatter(X(y==2, 1), X(y==2, 2), '*r')
scatter(X1(Z==1), X2(Z==1), '.b')
scatter(X1(Z==2), X2(Z==2), '.r')
hold off

