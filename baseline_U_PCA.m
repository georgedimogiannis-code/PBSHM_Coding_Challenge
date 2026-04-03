%--------------------------------------------------------------------------
%              Unsupervised PCA-based Method for Damage Detection (baseline phase)
%--------------------------------------------------------------------------
% This method uses centered PCA and comprises of two functions, one for the  
% baseline phase and a second one for the inspection phase. The present 
% function corresponds to the baseline phase and should be executed first. 
% Input variables:
%
% model_o : Feature matrix representing the healthy structures as
%           calculated in Task 3.m. It is noted that all models 
%           should have the same order n. Also p>>n, where p is the 
%           number of  healthy models available in the baseline phase.
%
% h:   Number of selected components.
%
% Output variables:
% PCs : Index of selected principal components. Inspection phase parameters 
%       are projected onto these components.
%
% coeff: Matrix containing all the principal components as column vectors.
%
% mYo: Sample mean vector of each structure's features available in this
%      phase.
%
% covarianceYo: Sample covariance matrix of all the baseline feature vectors.
%
% p:     No of models in the baseline phase           
%
% explained:  Percentage of variance explained by the first m components. 
%            This is user-selected.
%
% latent: Eigenvalues of the covariance matrix of the input data
%
% kk:     Condition number of covarianceYo 
%
% For this function to work, run Task3.m.

function [PCs,coeff,mYo,covarianceYo,p,explained,latent,kk] = baseline_U_PCA(model_o,h)

p = length(model_o); % Number of models in baseline phase
n = size(model_o,2); % Model order

Yo = zeros(p,n); % Initializing

for i = 1:p
    Yo(i,:)= model_o(i,:); % Gather the parameters of baseline models in a matrix
end

covarianceYo = cov(Yo); % Sample covariance of Yo

  kk=cond(covarianceYo);
[coeff,~,latent,~,explained,mYo] = pca(Yo,'Centered',true); % Centered PCA

PCs=1:h;
