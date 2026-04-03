 function [X_aug, y_aug] = SMOTE(X, y, minority_class, k, target_n)
% This function applies the Synthetic Minority Over-sampling Technique (SMOTE)
% to generate artificial samples for the minority class only (damaged), in 
% order to address class imbalance in the binary classification problem 
% postulated (35 Healthy/15 Damaged structures).
%
%Inputs:
%
% X            : N x d feature matrix, where each row is a sample and
%                each column is a feature.
% y            : N x 1 vector of class labels
%
% minority_class : Labels corresponding to the minority class(e.g. 1)
%
% k            : number of nearest neighbors
%
% target_n     : desired final number of minority samples after
%                augmentation
%
%Outputs:
%
%X_aug          : Augmented feature matrix containing:
%                  -original majority samples (0/Healthy)
%                  -original minority samples (1/Damaged)
%                  -synthetic minority samples
%
%y_aug          : Corresponding label vector for X_aug
%---------------------------------------------------------------------------------------------------------------

%Split of minority and majority class
X_min = X(y == minority_class, :);
X_maj = X(y ~= minority_class, :);
y_min = y(y == minority_class);
y_maj = y(y ~= minority_class);
n_min = size(X_min,1); %No of minority samples

%Check whether SMOTE is needed
if target_n <= n_min
    X_aug = X;
    y_aug = y;
    return
end

n_to_generate = target_n - n_min;

% if minority samples are very few, reduce k
k_eff = min(k, n_min - 1);
if k_eff < 1
    error('Not enough minority samples for SMOTE.');
end

%Find nearest neighbors
[idx, ~] = knnsearch(X_min, X_min, 'K', k_eff + 1);

%Generate synthetic samples
synthetic = zeros(n_to_generate, size(X,2));

for s = 1:n_to_generate
    i = randi(n_min);              % pick a minority sample
    nn_col = randi(k_eff) + 1;     % skip self at column 1
    
    x_i  = X_min(i,:);
    x_nn = X_min(idx(i,nn_col),:);
    
    lambda = rand;
    synthetic(s,:) = x_i + lambda * (x_nn - x_i); %SMOTE formula
end

X_aug = [X_maj; X_min; synthetic];
y_aug = [y_maj; y_min; repmat(minority_class, n_to_generate, 1)];
end
