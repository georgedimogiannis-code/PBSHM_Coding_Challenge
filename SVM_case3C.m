function [trainedClassifier, validationAccuracy, F1, AUC,XROC, YROC] = SVM_case3C(trainingData, responseData)
% Returns a trained SVM classifier and its validation accuracy, F1-score, AUC and ROC cuve data.
% Bayesian optimization is implemented for hyperparameter tuning.
%
%  Input:
%      trainingData: Feature matrix N_aug x 6, where N is the number of rows 
%      of the augmented feature matrix for the case of 6 storeys.
%      responseData: Healthy/Damaged label augmented output
%
%  Output:
%      trainedClassifier: Trained SVM classifier struct
%      validationAccuracy: Cross-validated accuracy
%      F1: F1-score
%      XROC: False positive rate values for ROC
%      YROC: True positive rate values for ROC
%      AUC: Area under ROC curve
%
% For this function to work, run Task2.m.
%
% This function is generated with the MATLAB Machine Learning Toolbox

%% Extract predictors and response
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6'};
predictors = inputTable(:, predictorNames);
response = responseData;

%% Train SVM classifier
classificationSVM = fitcsvm(...
    predictors, ...
    response, ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 19.55720229124587, ...
    'BoxConstraint', 986.583008908844, ...
    'Standardize', false, ...
    'ClassNames', [0; 1]);

%% Result struct 
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.ClassificationSVM = classificationSVM;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2022b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new predictor column matrix, X, use: \n  yfit = c.predictFcn(X) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nX must contain exactly 6 columns because this model was trained using 6 predictors. \nX must contain only predictor columns in exactly the same order and format as your training \ndata. Do not include the response column or any columns you did not import into the app. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

%% Cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 5);

%% Validation predictions and scores
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

%% Validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');

%% F1-score
% Assuming positive class = 1
TP = sum((validationPredictions == 1) & (response == 1));
FP = sum((validationPredictions == 1) & (response == 0));
FN = sum((validationPredictions == 0) & (response == 1));

precision = TP / (TP + FP);
recall    = TP / (TP + FN);

if (precision + recall) == 0
    F1 = 0;
else
    F1 = 2 * (precision * recall) / (precision + recall);
end

%% ROC and AUC
% Find the score column corresponding to positive class 1
classNames  = classificationSVM.ClassNames;            
posClassIdx = find(classNames == 1);
scorePositive = validationScores(:, posClassIdx);

[XROC, YROC, ~, AUC] = perfcurve(response, scorePositive, 1); 

end
