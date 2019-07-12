clear; clc;

data = readtable('../../data/exercises/food-texture.csv');

X = table2array(data(:,2:6));

%figure
%hold on
%scatter(X(:,1),X(:,2))
%hold off

k = 2;
clusters = kmeans(X, k);

inertias = [];

for k=1:10
    [clusters, centroids, inertia] = kmeans(X,k);
    inertias = [inertias; sum(inertia)];
end

figure
hold on
plot(inertias)
xlabel('Number of cluster')
ylabel('Inertia')
title('Elbow method')

% Choose a value
clusters = kmeans(X,4);

figure 
hold on
plot(X(clusters==1,1), X(clusters==1,2), '.r') % Oil vs Crispy space
plot(X(clusters==2,1), X(clusters==2,2), '.m')
plot(X(clusters==3,1), X(clusters==3,2), '.g')
plot(X(clusters==4,1), X(clusters==4,2), '.b')
xlabel('Oil')
ylabel('Density')
hold off

data.Clusters = clusters;


% Silhouette plot: similarity between points within the same cluster
clusters = kmeans(X,4);
silhouette(X, clusters)

randomClusters = randi([1,4],50,1);
silhouette(X, randomClusters)
