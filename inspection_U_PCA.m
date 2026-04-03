%--------------------------------------------------------------------------
%              Unsupervised PCA-based Method for Damage Detection (inspection phase)
%--------------------------------------------------------------------------
% This method uses centered PCA and comprises of two functions, one for the
% baseline phase and a second one for the inspection phase. The present
% function corresponds to the inspection phase and should be executed last.
%
% Input variables:
% model_u  : Feature matrix representing the healthy and damaged structures, 
%            as calculated in Task 3.m.
%
% PCs : Index of selected principal components. Inspection phase parameters
%       are projected onto these components.
%
% coeff: Matrix containing all the principal components as column vectors.
%
% mYo: Sample mean vector of each structure's features available in this
%      phase.
%
% covarianceYo: Sample covariance matrix of all the baseline feature
%               vectors.
%
% Statistic  : Choose between 'Eucl', for using Euclidean norm as the
%              method's test pseudo-statistic and 'Mahal' for using the
%              Mahalanobis norm as test pseudo-statistic.
%
% Output variable:
% D : The method’s test pseudo-statistic for the available baseline and
%     inspection phase models.
%
% For this function to work, run Task3.m.

function [D] = inspection_U_PCA(model_u,PCs,coeff,mYo,covarianceYo,Statistic)

Um = coeff(:,PCs); % Loading matrix defined based on the PCs selected in the baseline phase

theta_u = Um' * (model_u(:)-mYo(1,:)'); % PCA transformed feature vector
R = Um' * covarianceYo * Um; % PCA transformed covariance
% Test pseudo-statistic
switch Statistic
    case 'Eucl' % Sum of distances
        D = norm(theta_u); % Euclidean norm
        
    case 'Mahal' % Min of distances
        D = theta_u'*pinv(R)*theta_u; % Mahalanobis norm

    otherwise
        error('Unexpected pseudo-statistic type.')
end

end
