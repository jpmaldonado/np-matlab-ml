
clear;
fname = '../../data/exercises/FTIRSpectraInstantCoffee/FTIR_Spectra_instant_coffee.csv';
X = csvread(fname, 3, 1, [3 1 288 56]);
y = csvread(fname, 1, 1, [1 1 1 56]);
X = X';
y = y';

% EXERCISE: 
 figure 
 hold on
 plot(X(1,:), '.b')
 plot(X(37,:), 'dr')
 hold off

%%%
% Feature selection ==> Choose the more important variables
arabica = X(y==1, :);
robusta = X(y==2, :);

% ttest: Compare every variable to see if they come from same process

[H, P] = ttest2(arabica, robusta, 'Vartype', 'unequal');

figure
hold on
ecdf(P)
xlabel('P value')
ylabel('CDF value')
hold off

% P-values close to zero = useful variables
[~, sortedFeatures ] = sort(P,2);


svm = fitcsvm(X(:, sortedFeatures(1:20)), y);
cv = crossval(svm);
loss = kfoldLoss(cv, 'mode', 'individual');
hist(loss)

