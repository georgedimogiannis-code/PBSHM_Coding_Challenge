function [X,Y,AUC]=ROC_CURVES(ValueH,ValueD)
% Calculating the X and Y values for the ROC plots as well as the AUC
% values for the damaged cases
%
% Inputs:
% ValueH ---> contains the distances of the healthy state
%
% ValueD ---> contains the distances of the damaged state
%
% Outputs:
%
% X ---> the X axis values for the ROC curves of the damaged cases
% Y ---> the Y axis values for the ROC curves of the damaged cases
% AUC ---> the AUC values of the damaged cases
%
% For this function to work, run Task3.m.

Healthy=ValueH';
Faulty=ValueD';
LH=length(Healthy);LF=length(Faulty);
for i=1:LH
    Labels(i,1) = 0;
    Scores(i,1) = Healthy(1,i);
end
for i=LH+1:LH+LF
    Labels(i,1) = 1;
    Scores(i,1) = Faulty(1,i-LH);
end

% Contains the X and Y of the ROC curve and the value of the AUC
[X,Y,~,AUC] = perfcurve(Labels,Scores,1,'XVals',0:1/4000:1);

