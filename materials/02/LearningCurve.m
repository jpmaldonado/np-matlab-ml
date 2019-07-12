data = csvread('../../data/demo/nlsdata.csv');
X = data(:, 1:2);
y = data(:, 3);
rng(42) % Random number generator seed

cvp = cvpartition(y, 'KFold',10);

trainErrors = [];
testErrors = [];
XtrainPrev = [];
XtestPrev = [];
ytrainPrev = [];
ytestPrev = [];
for i = 1:cvp.NumTestSets
    XtrainPrev = [XtrainPrev; X(cvp.training(i), :)]; % Indices for training in i-th fold
    ytrainPrev = [ytrainPrev; y(cvp.training(i))];
    XtestPrev = [XtestPrev; X(cvp.test(i), :)]; 
    ytestPrev = [ytestPrev; y(cvp.test(i))];
    
    svm = fitcsvm(XtrainPrev, ytrainPrev, 'KernelFunction', 'gaussian'); % Train model
    
    % keeping training error
    ypredsTrain = predict(svm, XtrainPrev);
    lossTrain = sum(ypredsTrain~=ytrainPrev)/numel(ytrainPrev);
    trainErrors = [trainErrors; lossTrain];
    
    % keeping test error
    ypredsTest = predict(svm, XtestPrev);
    lossTest = sum(ypredsTest~=ytestPrev)/numel(ytestPrev);
    testErrors = [testErrors; lossTest];
end

numExamples = 86*(1:10);
figure 
hold on
plot(numExamples, trainErrors, '-b')
plot(numExamples, testErrors, '--r')
ylabel('Error')
xlabel('Number of observations')
legend('Train Error', 'Test Error')
hold off



