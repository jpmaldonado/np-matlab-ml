data = readtable('../data/exercises/cheddar-cheese.csv');

X = data(:, {'Acetic', 'H2S', 'Lactic'});
y = data(:, 'Taste');

svr = fitrsvm(X, y, 'KernelFunction', 'polynomial', 'PolynomialOrder', 2);