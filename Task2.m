%--------------------------------------------------------------------------
%------------------------------------TASK 2 solution------------------------
%--------------------------------------------------------------------------
%This script implements the solution of TASK 2 for the PBSHM Coding Challenge.
%A simple baseline is built for damage detection using fixed-length summaries
%of each structure. The SVM classifier is considered, using an equal No of 
%healthy/damaged members for its proper function. The following feature
%matrices are considered:
%
%CASE 1: Feature matrix N x MIN_DOF employing the dominant modal frequencies 
%        of the first 4 storeys of the population, (N=30, MIN_DOF=4).
%
%CASE 2: Feature matrix N x (mean(f), std(f), skewness(f), mean(h), std(h), 
%        sum(h), n_storeys) including statistical quantities of frequency
%        and geometry features (f: frequency, h: height)
%
%CASE 3: Distinct feature matrices for members with storeys 4-8. Before
%        training, artificial samples are generated with the SMOTE
%        technique in order to address the class imbalance problem.
%
%Appropriate metrics, such as accuracy, F1, and ROC-AUC using a 5-fold cross-
% validation, are extracted to estimate detection performance. 

clear
clc
close all

measurements = jsondecode(fileread('structures_measurements.json'));
labels=readtable('structure_labels.csv');

N=50; %No of structures
N_HEALTHY=35; %No of healthy structures
N_DAMAGED=15; %No of damaged structures
MAX_DOF=8;    %Maximum No of storeys
MIN_DOF=4;    %Minimum No of storeys
n_storeys=[measurements.n_storeys]'; %No of Storeys per Structure

%Initialization of matrices 
total_height=zeros(N,1);
mean_height=zeros(N,1);
std_height=zeros(N,1);
mean_f=zeros(N,1);
std_f=zeros(N,1);
skewness_f=zeros(N,1);

for i=1:N
  total_height(i) = sum([measurements(i).node_features.height_m]);
  mean_height(i) = mean([measurements(i).node_features.height_m]);
  std_height(i) = std([measurements(i).node_features.height_m]);
end

for i=1:N
  mean_f(i) = mean([measurements(i).node_features.dominant_modal_frequency_Hz]);
  std_f(i) = std([measurements(i).node_features.dominant_modal_frequency_Hz]);
  skewness_f(i) = skewness([measurements(i).node_features.dominant_modal_frequency_Hz]);
end

%Matrix of dominant modal frequencies, padded with zeros
frequencies= zeros(N,MAX_DOF);

for i=1:N
    X= [measurements(i).node_features.dominant_modal_frequency_Hz]';
    n=length(X);
    frequencies(i,1:n)=X;
end

%Matrices of the frequencies of the first 4 storeys of healthy/damaged
first_4_frequencies_healthy= zeros(N_HEALTHY,MIN_DOF);
first_4_frequencies_damaged= zeros(N_DAMAGED,MIN_DOF);

j=0; k=0;  %Row counters 

for i=1:N

if labels.damaged(i)==0

    j=j+1;
    first_4_frequencies_healthy(j,:)= frequencies(i,1:MIN_DOF);

elseif labels.damaged(i)==1
   
    k=k+1;
    first_4_frequencies_damaged(k,:)= frequencies(i,1:MIN_DOF);
   
end

end


%Label healthy/damaged cases
y= zeros(N,1);

for i=1:N
    y(i)=labels.damaged(i);
end

%Healthy/Damage indices
healthy_idx= zeros(N_HEALTHY,1);
damage_idx= zeros(N_DAMAGED,1);

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

%CASE 1
FEATURE1=[first_4_frequencies_healthy(1:N_DAMAGED,:); first_4_frequencies_damaged];

%CASE 2
F2=[mean_f std_f skewness_f mean_height std_height total_height n_storeys];  
FEATURE2_H=zeros(N_DAMAGED,size(F2,2));
FEATURE2_F=zeros(N_DAMAGED,size(F2,2));

for i=1:N_DAMAGED
FEATURE2_H(i,:)= F2(healthy_idx(i),:);
end

for i=1:N_DAMAGED
FEATURE2_F(i,:)= F2(damage_idx(i),:);
end

FEATURE2=[FEATURE2_H; FEATURE2_F]; %Reduction to 15 Healthy/15 Damage cases

Y=[zeros(15,1); ones(15,1)];  %Output matrix with structure-level label

%CASE3

count1=0; %Counter of members with 4 storeys
count2=0; %Counter of members with 5 storeys
count3=0; %Counter of members with 6 storeys
count4=0; %Counter of members with 7 storeys
count5=0; %Counter of members with 8 storeys

for i=1:N
    if n_storeys(i)==4
        count1=count1+1;
    elseif n_storeys(i)==5
        count2=count2+1;
    elseif n_storeys(i)==6
        count3=count3+1;
    elseif n_storeys(i)==7
        count4=count4+1;
    elseif n_storeys(i)==8
        count5=count5+1;
    end
end

%Frequencies/No of storeys, corresponding labels and indices
f4= zeros(count1,4); y4= zeros(count1,1); a=0; 
f5= zeros(count2,5); y5= zeros(count2,1); b=0;
f6= zeros(count3,6); y6= zeros(count3,1); c=0;
f7= zeros(count4,7); y7= zeros(count4,1); d=0;
f8= zeros(count5,8); y8= zeros(count5,1); e=0;

for i=1:N
    if measurements(i).n_storeys==4
        a=a+1;
        f4(a,:)=frequencies(i,1:4);

         if y(i)==0
             y4(a)=0;
         else  
             y4(a)=1;

         end
    end
end

for i=1:N
    if measurements(i).n_storeys==5
        b=b+1;
        f5(b,:)=frequencies(i,1:5);

         if y(i)==0
             y5(b)=0;
         else  
             y5(b)=1;

         end
    end
end


for i=1:N
    if measurements(i).n_storeys==6
        c=c+1;
        f6(c,:)=frequencies(i,1:6);

         if y(i)==0
             y6(c)=0;
         else  
             y6(c)=1;

         end
    end
end

for i=1:N
    if measurements(i).n_storeys==7
        d=d+1;
        f7(d,:)=frequencies(i,1:7);

         if y(i)==0
             y7(d)=0;
         else 
             y7(d)=1;

         end
    end
end

for i=1:N
    if measurements(i).n_storeys==8
        e=e+1;
        f8(e,:)=frequencies(i,1:8);

         if y(i)==0
             y8(e)=0;
         else  
             y8(e)=1;

         end
    end
end

%Application of SMOTE-Call of SMOTE function
[X_aug4, y_aug4] = SMOTE(f4, y4, 1, 2, sum(y4==0));
[X_aug5, y_aug5] = SMOTE(f5, y5, 1, 2, sum(y5==0));
[X_aug6, y_aug6] = SMOTE(f6, y6, 1, 2, sum(y6==0));
[X_aug7, y_aug7] = SMOTE(f7, y7, 1, 2, sum(y7==0));
[X_aug8, y_aug8] = SMOTE(f8, y8, 1, 2, sum(y8==0));

 %% ----------------------------------------FIGURES----------------------------------------------------------------

%CASE 1
%Call of SVM_case1 function
[trainedClassifier, validationAccuracy, F1, AUC,XROC, YROC] = SVM_case1(FEATURE1, Y);

figure('Color','w')
sgtitle('\textbf{Damage detection performance of SVM: Case 1}')

subplot(1,2,1)
bar([validationAccuracy F1 AUC])
title('\textbf{Accuracy, F1 and AUC metrics}')
xticklabels({'Accuracy','F1','AUC'}), ylim([0 1])

subplot(1,2,2)
plot(XROC,YROC,'Linewidth',1.4)
xlabel('False Positive Rate'), ylabel('True Positive Rate')
title('\textbf{ROC curve}')
set(findall(gcf,'-property','FontSize'),'FontSize',14)
set(findall(gcf,'-property','FontSize'),'FontSize',14)
set(findall(gcf,'-property','Interpreter'),'Interpreter','Latex')
set(findall(gcf,'-property','TickLabelInterpreter'),'TickLabelInterpreter','Latex')

%CASE 2
%Call of SVM_case2 function
[trainedClassifier, validationAccuracy, F1, AUC,XROC, YROC] = SVM_case2(FEATURE2, Y);

figure('Color','w')
sgtitle('\textbf{Damage detection performance of SVM: Case 2}')

subplot(1,2,1)
bar([validationAccuracy F1 AUC])
title('\textbf{Accuracy, F1 and AUC metrics}')
xticklabels({'Accuracy','F1','AUC'}), ylim([0 1])

subplot(1,2,2)
plot(XROC,YROC,'Linewidth',1.4)
xlabel('False Positive Rate'), ylabel('True Positive Rate')
title('\textbf{ROC curve}')
set(findall(gcf,'-property','FontSize'),'FontSize',14)
set(findall(gcf,'-property','FontSize'),'FontSize',14)
set(findall(gcf,'-property','Interpreter'),'Interpreter','Latex')
set(findall(gcf,'-property','TickLabelInterpreter'),'TickLabelInterpreter','Latex')

%CASE 3A: Feature matrix of four storeys
%Call of SVM_case3A function
[trainedClassifier, validationAccuracy, F1, AUC,XROC, YROC] = SVM_case3A(X_aug4, y_aug4);

figure('Color','w')
sgtitle('\textbf{Damage detection performance of SVM: Case 3 (4 storeys)}')

subplot(1,2,1)
bar([validationAccuracy F1 AUC])
title('\textbf{Accuracy, F1 and AUC metrics}')
xticklabels({'Accuracy','F1','AUC'}), ylim([0 1])

subplot(1,2,2)
plot(XROC,YROC,'Linewidth',1.4)
xlabel('False Positive Rate'), ylabel('True Positive Rate')
title('\textbf{ROC curve}')
set(findall(gcf,'-property','FontSize'),'FontSize',14)
set(findall(gcf,'-property','FontSize'),'FontSize',14)
set(findall(gcf,'-property','Interpreter'),'Interpreter','Latex')
set(findall(gcf,'-property','TickLabelInterpreter'),'TickLabelInterpreter','Latex')

%CASE 3B: Feature matrix of five storeys
%Call of SVM_case3B function
[trainedClassifier, validationAccuracy, F1, AUC,XROC, YROC] = SVM_case3B(X_aug5, y_aug5);

figure('Color','w')
sgtitle('\textbf{Damage detection performance of SVM: Case 3 (5 storeys)}')

subplot(1,2,1)
bar([validationAccuracy F1 AUC])
title('\textbf{Accuracy, F1 and AUC metrics}')
xticklabels({'Accuracy','F1','AUC'}), ylim([0 1])

subplot(1,2,2)
plot(XROC,YROC,'Linewidth',1.4)
xlabel('False Positive Rate'), ylabel('True Positive Rate')
title('\textbf{ROC curve}')
set(findall(gcf,'-property','FontSize'),'FontSize',14)
set(findall(gcf,'-property','FontSize'),'FontSize',14)
set(findall(gcf,'-property','Interpreter'),'Interpreter','Latex')
set(findall(gcf,'-property','TickLabelInterpreter'),'TickLabelInterpreter','Latex')

%CASE 3C: Feature matrix of six storeys
%Call of SVM_case3C function
[trainedClassifier, validationAccuracy, F1, AUC,XROC, YROC] = SVM_case3C(X_aug6, y_aug6);

figure('Color','w')
sgtitle('\textbf{Damage detection performance of SVM: Case 3 (6 storeys)}')

subplot(1,2,1)
bar([validationAccuracy F1 AUC])
title('\textbf{Accuracy, F1 and AUC metrics}')
xticklabels({'Accuracy','F1','AUC'}), ylim([0 1])

subplot(1,2,2)
plot(XROC,YROC,'Linewidth',1.4)
xlabel('False Positive Rate'), ylabel('True Positive Rate')
title('\textbf{ROC curve}')
set(findall(gcf,'-property','FontSize'),'FontSize',14)
set(findall(gcf,'-property','FontSize'),'FontSize',14)
set(findall(gcf,'-property','Interpreter'),'Interpreter','Latex')
set(findall(gcf,'-property','TickLabelInterpreter'),'TickLabelInterpreter','Latex')

%CASE 3D: Feature matrix of seven storeys
%Call of SVM_case3D function
[trainedClassifier, validationAccuracy, F1, AUC,XROC, YROC] = SVM_case3D(X_aug7, y_aug7);

figure('Color','w')
sgtitle('\textbf{Damage detection performance of SVM: Case 3 (7 storeys)}')

subplot(1,2,1)
bar([validationAccuracy F1 AUC])
title('\textbf{Accuracy, F1 and AUC metrics}')
xticklabels({'Accuracy','F1','AUC'}), ylim([0 1])

subplot(1,2,2)
plot(XROC,YROC,'Linewidth',1.4)
xlabel('False Positive Rate'), ylabel('True Positive Rate')
title('\textbf{ROC curve}')
set(findall(gcf,'-property','FontSize'),'FontSize',14)
set(findall(gcf,'-property','FontSize'),'FontSize',14)
set(findall(gcf,'-property','Interpreter'),'Interpreter','Latex')
set(findall(gcf,'-property','TickLabelInterpreter'),'TickLabelInterpreter','Latex')

%CASE 3E: Feature matrix of eight storeys
%Call of SVM_case3E function
[trainedClassifier, validationAccuracy, F1, AUC,XROC, YROC] = SVM_case3E(X_aug8, y_aug8);

figure('Color','w')
sgtitle('\textbf{Damage detection performance of SVM: Case 3 (8 storeys)}')

subplot(1,2,1)
bar([validationAccuracy F1 AUC])
title('\textbf{Accuracy, F1 and AUC metrics}')
xticklabels({'Accuracy','F1','AUC'}), ylim([0 1])

subplot(1,2,2)
plot(XROC,YROC,'Linewidth',1.4)
xlabel('False Positive Rate'), ylabel('True Positive Rate')
title('\textbf{ROC curve}')
set(findall(gcf,'-property','FontSize'),'FontSize',14)
set(findall(gcf,'-property','FontSize'),'FontSize',14)
set(findall(gcf,'-property','Interpreter'),'Interpreter','Latex')
set(findall(gcf,'-property','TickLabelInterpreter'),'TickLabelInterpreter','Latex')
