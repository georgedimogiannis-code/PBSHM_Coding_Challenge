%--------------------------------------------------------------------------
%------------------------------------TASK 1 solution------------------------
%--------------------------------------------------------------------------
%This script implements the solution of TASK 1 for the PBSHM Coding Challenge.
%The population of 50 shear-frame structures is explored within proper
%bar plots, histograms and box plots in order to understand the variation
%across the structures.

clear
close all
clc

N=50;  %No of structures
N_HEALTHY=35;  %Healthy structures
N_DAMAGED=15;  %Damaged structures
MIN_DOF=4;     %Minimun No of storeys
MAX_DOF=8;     %Maximum No of storeys

%Load files
measurements = jsondecode(fileread('structures_measurements.json'));
labels= readtable("structure_labels.csv");
edges_weights= readtable('population_edge_weights_geometry.csv');

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


%% -----------------------------------------FIGURES-----------------------------------------------------------------

%Bar plot of No of Storeys per Structure and Total Heights
figure('Color','White')
subplot(1,2,1)
bar(n_storeys)
xlabel('Structure ID'), ylabel('No of Storeys')

subplot(1,2,2)
bar(total_height)
xlabel('Structure ID'), ylabel('Total Height (m)')
set(findall(gcf,'-property','FontSize'),'FontSize',12)
set(findall(gcf,'-property','FontSize'),'FontSize',12)
set(findall(gcf,'-property','Interpreter'),'Interpreter','Latex')
set(findall(gcf,'-property','TickLabelInterpreter'),'TickLabelInterpreter','Latex')

%Bar plot of average storey heights and std of storey heights
figure('Color','White')
subplot(1,2,1)
bar(mean_height)
xlabel('Structure ID'), ylabel('Average Storey Height (m)')

subplot(1,2,2)
bar(std_height)
xlabel('Structure ID'), ylabel('Standard Deviation in Storey Heights')
set(findall(gcf,'-property','FontSize'),'FontSize',12)
set(findall(gcf,'-property','FontSize'),'FontSize',12)
set(findall(gcf,'-property','Interpreter'),'Interpreter','Latex')
set(findall(gcf,'-property','TickLabelInterpreter'),'TickLabelInterpreter','Latex')

%Histogram of storeys and total height
figure('Color','White')
subplot(1,2,1)
histogram(n_storeys)
xlabel('No of Storeys'), ylabel('No of Structures')
sgtitle('\textbf{Histograms of number of storeys and total height}')

subplot(1,2,2)
histogram(total_height,10) 
xlabel('Total Height (m)'), ylabel('No of Structures')
set(findall(gcf,'-property','FontSize'),'FontSize',12)
set(findall(gcf,'-property','FontSize'),'FontSize',12)
set(findall(gcf,'-property','Interpreter'),'Interpreter','Latex')
set(findall(gcf,'-property','TickLabelInterpreter'),'TickLabelInterpreter','Latex')

%Construction of undirected similarity graph
s = edges_weights.source + 1;   
t = edges_weights.target + 1;   
w = edges_weights.cosine_similarity;

G = graph(s, t, w);

figure('Color','White')
p=plot(G, 'Layout', 'force');
title('\textbf{Population Similarity Graph}')

%Define node colors
numNodes = numnodes(G);
nodeColors = zeros(numNodes,3);

% Blue = healthy (0), Red = damaged (1)
nodeColors(labels.damaged == 0, :) = repmat([0 0.4470 0.7410], sum(labels.damaged==0), 1);
nodeColors(labels.damaged == 1, :) = repmat([0.8500 0.3250 0.0980], sum(labels.damaged==1), 1);

%Apply colors
p.NodeColor = nodeColors;
p.MarkerSize = 9; hold on
p.NodeFontSize = 13;
h1 = plot(nan, nan, 'o', 'MarkerFaceColor', [0 0.4470 0.7410], 'MarkerEdgeColor','none');
h2 = plot(nan, nan, 'o', 'MarkerFaceColor', [0.8500 0.3250 0.0980], 'MarkerEdgeColor','none');
legend([h1 h2], {'Healthy','Damaged'},'Location','southeast')

set(findall(gcf,'-property','FontSize'),'FontSize',15)
set(findall(gcf,'-property','FontSize'),'FontSize',15)
set(findall(gcf,'-property','Interpreter'),'Interpreter','Latex')
set(findall(gcf,'-property','TickLabelInterpreter'),'TickLabelInterpreter','Latex')

%Bar plot of the first 4 frequencies
figure('Color','White')
bar(1:MIN_DOF, frequencies(:,1:MIN_DOF))
xlabel('No of Storey'), ylabel('Dominant Modal Frequency (Hz)')
title('\textbf{Bar plot of the dominant modal frequencies of the first 4 storeys}')
set(findall(gcf,'-property','FontSize'),'FontSize',12)
set(findall(gcf,'-property','FontSize'),'FontSize',12)
set(findall(gcf,'-property','Interpreter'),'Interpreter','Latex')
set(findall(gcf,'-property','TickLabelInterpreter'),'TickLabelInterpreter','Latex')

%Box plot of the frequencies of the first 4 storeys of healthy/damaged
figure('Color','White')

titles = {'\textbf{Storey 1}','\textbf{Storey 2}','\textbf{Storey 3}','\textbf{Storey 4}'};
for i = 1:MIN_DOF
    subplot(1,MIN_DOF,i)
   
    data = [first_4_frequencies_healthy(:,i); first_4_frequencies_damaged(:,i)];
    group = [repmat({'Healthy'}, N_HEALTHY, 1); ... % Create group labels
             repmat({'Damaged'}, N_DAMAGED, 1)];
    
    boxplot(data, group)
    title(titles{i}), ylabel('Frequency (Hz)')
    grid on
   
end

set(findall(gcf,'-property','FontSize'),'FontSize',12)
set(findall(gcf,'-property','FontSize'),'FontSize',12)
set(findall(gcf,'-property','Interpreter'),'Interpreter','Latex')
set(findall(gcf,'-property','TickLabelInterpreter'),'TickLabelInterpreter','Latex')
