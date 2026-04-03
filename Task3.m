%--------------------------------------------------------------------------
%------------------------------------TASK 3 solution------------------------
%--------------------------------------------------------------------------
%This script implements the solution of TASK 3 for the PBSHM Coding Challenge.
%A simple unsupervised PCA is considered.
%Two cases of feature matrices are considered:
%
%CASE 1: Feature matrix N x 4 with the population's most dominant modal 
%        frequencies of the first 4 storeys (N=50).     
%
%CASE 2: Feature matrix N x 7 with the mean, std, and skewness of the dominant
%        modal frequencies and a summary of geometric features, including
%        the sum, stdm and mean of each member's storey heights and the
%        number of storeys.
%
%The detection is based on Mahalanobis distance metric plots and
%corresponding ROC-AUC plots. The eigenvalues of the sample covariance 
% matrices, the fraction of total variance explained per PC, and the
%cumulative sum fraction of total variance explained are also depicted.
%The 60% of the total population is used for the baseline phase (healthy ONLY)
%and the rest for the inspection phase (10% Healthy/30% Damaged).

clear
clc
close all

measurements = jsondecode(fileread('structures_measurements.json'));
labels= readtable('structure_labels.csv');

%Parameters
N=50;  %No of structures
N_baseline= 30; %No of baseline observations
N_inspection=20; %No of inspection observations
N_healthy=35; %No of healthy members
N_damaged= 15; %No of damaged members
min_dof=4; %Minimum No of storeys
max_dof=8; %Maximum No of storeys
geometric_features= 4; %Geometric features used in the feature matrix of case 2

%Matrix of the dominant modal frequencies of the population for the first
%4 storeys
frequencies= zeros(N,min_dof);

for i=1:N
    X= [measurements(i).node_features.dominant_modal_frequency_Hz]';
    frequencies(i,:)=X(1:min_dof);
end

n_storeys=[measurements.n_storeys]'; %No of Storeys per Structure

%Initialization of matrices of total height, mean height, and std of height
total_height=zeros(N,1);
mean_height=zeros(N,1);
std_height=zeros(N,1);

for i=1:N
  total_height(i) = sum([measurements(i).node_features.height_m]);
  mean_height(i) = mean([measurements(i).node_features.height_m]);
  std_height(i) = std([measurements(i).node_features.height_m]);
end

%Initialization of matrices of mean, std, and skewness of frequencies
mean_f=zeros(N,1);
std_f=zeros(N,1);
skewness_f=zeros(N,1);

for i=1:N
  mean_f(i) = mean([measurements(i).node_features.dominant_modal_frequency_Hz]);
  std_f(i) = std([measurements(i).node_features.dominant_modal_frequency_Hz]);
  skewness_f(i) = skewness([measurements(i).node_features.dominant_modal_frequency_Hz]);
end

PCA_FEATURE1=frequencies;
PCA_FEATURE2=[mean_f std_f skewness_f mean_height std_height total_height n_storeys];  

%% CASE 1

%Label healthy/damaged cases
y= zeros(N,1);

for i=1:N
    y(i)=labels.damaged(i);
end

%Healthy/Damage indices
healthy_idx= zeros(N_healthy,1);
damage_idx= zeros(N_damaged,1);

j=0; k=0; %Counters

for i=1:N
    if y(i)==0
        j=j+1;
        healthy_idx(j)=i;
    else
        k=k+1;
        damage_idx(k)=i;
    end
end

baseline1=zeros(N_baseline,min_dof); %Initializing

for i=1:N_baseline
    baseline1(i,:)=PCA_FEATURE1(healthy_idx(i),:);
end

inspection_healthy1= zeros(N_healthy-N_baseline,min_dof);
inspection_damaged1= zeros(N_damaged,min_dof);

for i=1:N_healthy-N_baseline
    inspection_healthy1(i,:)=PCA_FEATURE1(healthy_idx(i+30),:);
   
end

for i=1:N_damaged
inspection_damaged1(i,:)= PCA_FEATURE1(damage_idx(i),:);
end

inspection1=[inspection_healthy1; inspection_damaged1];

%Initializing the Mahalanobis distances
D1= zeros(N_inspection,1); 

PCs1=1;  %No of remaining components

%Call of baseline_U_PCA function 
[PCs,coeff,mYo,covarianceYo,p,explained,latent,kk] = baseline_U_PCA(baseline1,PCs1);

%Call of inspection_U_PCA function 
for i=1:N_inspection
D1(i) = inspection_U_PCA(inspection1(i,:),PCs,coeff,mYo,covarianceYo,'Mahal');
end

gamma1=explained; %Fraction of total variance explained
eigen1=latent;  %Eigenvalues of the sample covariance matrix

cumExplained1 = cumsum(gamma1); %Cumulative sum of explained variance
pc1 = 1:length(gamma1);  %Total No of PCs

%% CASE 2

baseline2=zeros(N_baseline,7); 

for i=1:N_baseline
    baseline2(i,:)=PCA_FEATURE2(healthy_idx(i),:);
end

inspection_healthy2= zeros(N_healthy-N_baseline,7);
inspection_damaged2= zeros(N_damaged,7);

for i=1:N_healthy-N_baseline
    inspection_healthy2(i,:)=PCA_FEATURE2(healthy_idx(i+30),:);
   
end

for i=1:N_damaged
inspection_damaged2(i,:)= PCA_FEATURE2(damage_idx(i),:);
end

inspection2=[inspection_healthy2; inspection_damaged2];

D2= zeros(N_inspection,1); 

PCs2=6; %No of remaining components

%Call of baseline_U_PCA function
[PCs,coeff,mYo,covarianceYo,p,explained,latent,kk] = baseline_U_PCA(baseline2,PCs2);

%Call of inspection_U_PCA function
for i=1:N_inspection
D2(i) = inspection_U_PCA(inspection2(i,:),PCs,coeff,mYo,covarianceYo,'Mahal');
end

gamma2= explained;
eigen2=latent;

cumExplained2 = cumsum(gamma2);
pc2 = 1:length(gamma2);

%% ---------------------------FIGURES------------------------------------

%Case1
%Plot of fraction of total variance explained and cumulative sum fraction
figure('Color','w')
yyaxis left
bar(pc1, gamma1)
ylabel('Fraction of total variance explained per principal component (%)')

yyaxis right
plot(pc1, cumExplained1, '-o', 'LineWidth', 1.2, 'MarkerSize', 3)
ylabel('Cumulative sum fraction of total variance explained (%)')

xlabel('Principal component')
title('\textbf{Explained fraction $\gamma$ of the total parameter variability: Case 1}')

xlim([0.5 length(gamma1)+0.5]),  ylim([0 max(gamma1)*1.15])

yyaxis right
ylim([30 100])

set(findall(gcf,'-property','FontSize'),'FontSize',12)
set(findall(gcf,'-property','FontSize'),'FontSize',12)
set(findall(gcf,'-property','Interpreter'),'Interpreter','Latex')
set(findall(gcf,'-property','TickLabelInterpreter'),'TickLabelInterpreter','Latex')

%Plot of the eigenfrequencies of the covariance matrix: Case 1 and 2
figure('Color','w')
subplot(1,2,1)
plot(eigen1,'o', ...
    'MarkerFaceColor', [0 0.4470 0.7410], ...   % blue fill
    'MarkerEdgeColor', 'k')                     % black edge 
xlabel('Principal component'), ylabel('Eigenvalues')
xlim([1 min_dof])
title('\textbf{Eigenvalues of sample covariance matrix: Case 1}')
set(gca, 'YScale', 'log')

subplot(1,2,2)
plot(eigen2,'o', ...
    'MarkerFaceColor', [0 0.4470 0.7410], ...   % blue fill
    'MarkerEdgeColor', 'k')                     % black edge
xlabel('Principal component'), ylabel('Eigenvalues')
xlim([1 7])
title('\textbf{Eigenvalues of sample covariance matrix: Case 2}')

set(gca, 'YScale', 'log')
set(findall(gcf,'-property','FontSize'),'FontSize',14)
set(findall(gcf,'-property','FontSize'),'FontSize',14)
set(findall(gcf,'-property','Interpreter'),'Interpreter','Latex')
set(findall(gcf,'-property','TickLabelInterpreter'),'TickLabelInterpreter','Latex')

%Case 2
%Plot of fraction of explained total variance and cumulative sum fraction
figure('Color','w')
yyaxis left
bar(pc2, gamma2)
ylabel('Fraction of total variance explained per principal component (%)')

yyaxis right
plot(pc2, cumExplained2, '-o', 'LineWidth', 1.2, 'MarkerSize', 3)
ylabel('Cumulative sum fraction of total variance explained (%)')
xlabel('Principal component')
title('\textbf{Explained fraction $\gamma$ of the total parameter variability: Case 2}')
ylim([30 100]),xlim([0.5 length(gamma2)+0.5])

set(findall(gcf,'-property','FontSize'),'FontSize',12)
set(findall(gcf,'-property','FontSize'),'FontSize',12)
set(findall(gcf,'-property','Interpreter'),'Interpreter','Latex')
set(findall(gcf,'-property','TickLabelInterpreter'),'TickLabelInterpreter','Latex')

%Cases 1,2: Extraction of distance metric plots and ROC curves

x = (1:N_inspection)'; %No of values in the x-axis of ROC curve
%Mahalanobis distances of healthy/damaged
y1 = [D1(1:N_healthy-N_baseline);D1(N_healthy-N_baseline+1:end)]; 
y2 = [D2(1:N_healthy-N_baseline);D2(N_healthy-N_baseline+1:end)]; 
groups = [ones(N_healthy-N_baseline, 1); 2 * ones(N_damaged, 1)]; % Clusters

%Call of ROC_CURVES function
[X1,Y1,AUC1]=ROC_CURVES(D1(1:N_healthy-N_baseline), D1(N_healthy-N_baseline+1:end));
[X2,Y2,AUC2]=ROC_CURVES(D2(1:N_healthy-N_baseline), D2(N_healthy-N_baseline+1:end));

%Case 1
figure('Color','White')
sgtitle('\textbf{Unsupervised PCA-based damage detection: Case 1}')
subplot(1,2,1)
hold on, box on
colors = lines(max(groups)); % Use a colormap (e.g., 'lines', 'parula', etc.)
for g = unique(groups)'
    scatter(x(groups == g), y1(groups == g), 50, colors(g, :), 'filled','MarkerEdgeColor', 'k');
end
hold off;

% Logarithmic y-axis
set(gca, 'YScale', 'log')
legend({'Healthy', 'Damaged'}, 'Location', 'best')
% Labels and title
xlabel('Inspection Case'), ylabel('Mahalanobis Distance')
title('{Distance Metric Plot}')

subplot(1,2,2)
plot(X1,Y1,'Linewidth',1.4)
xlabel('False positive rate'), ylabel('True positive rate')
title('\textbf{ROC Curves}')
set(findall(gcf,'-property','FontSize'),'FontSize',12)
set(findall(gcf,'-property','FontSize'),'FontSize',12)
set(findall(gcf,'-property','Interpreter'),'Interpreter','Latex')
set(findall(gcf,'-property','TickLabelInterpreter'),'TickLabelInterpreter','Latex')

%Case 2
figure('Color','White')
sgtitle('\textbf{Unsupervised PCA-based damage detection: Case 2}')
subplot(1,2,1)
hold on, box on
colors = lines(max(groups)); % Use a colormap (e.g., 'lines', 'parula', etc.)
for g = unique(groups)'
    scatter(x(groups == g), y2(groups == g), 50, colors(g, :), 'filled','MarkerEdgeColor', 'k');
end
hold off;

% Logarithmic y-axis
set(gca, 'YScale', 'log')
legend({'Healthy', 'Damaged'}, 'Location', 'best')
% Labels and title
xlabel('Inspection Case'), ylabel('Mahalanobis Distance')
title('\textbf{Distance Metric Plot}')
set(findall(gcf,'-property','FontSize'),'FontSize',12)
set(findall(gcf,'-property','FontSize'),'FontSize',12)
set(findall(gcf,'-property','Interpreter'),'Interpreter','Latex')
set(findall(gcf,'-property','TickLabelInterpreter'),'TickLabelInterpreter','Latex')

subplot(1,2,2)
plot(X2,Y2,'Linewidth',1.4)
xlabel('False positive rate'), ylabel('True positive rate')
title('\textbf{ROC Curves}')
set(findall(gcf,'-property','FontSize'),'FontSize',12)
set(findall(gcf,'-property','FontSize'),'FontSize',12)
set(findall(gcf,'-property','Interpreter'),'Interpreter','Latex')
set(findall(gcf,'-property','TickLabelInterpreter'),'TickLabelInterpreter','Latex')

%Bar plots of AUC's 
figure('Color','White')
bar([AUC1 AUC2])
ylabel('AUC')
title('\textbf{AUC for PCA-based damage detection methods}')
ylim([0 1])
xticklabels({'Case 1','Case 2'})

set(findall(gcf,'-property','FontSize'),'FontSize',12)
set(findall(gcf,'-property','FontSize'),'FontSize',12)
set(findall(gcf,'-property','Interpreter'),'Interpreter','Latex')
set(findall(gcf,'-property','TickLabelInterpreter'),'TickLabelInterpreter','Latex')
