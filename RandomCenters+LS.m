clc;clear all;close all;

load('Train.mat');
load('GroundTruth.mat');
load('Test.mat');
load('TrueClass.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% FINDING CENTERS
N = 60000;
K = 100;
indices = randperm(60000, K);
centres_new = Train(indices, :);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% FINDING VARIANCES
d_max = 0;
for i = 1:K-1
    my_dist(i) = sum((centres_new(i,:)-centres_new(i+1,:)).^2);
    if  my_dist(i) > d_max
            d_max = my_dist(i);
    else
    end
end

% Normalization method from the slides
spread = (d_max/sqrt(2*K))*ones(K,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% FINDING WEIGHTS
centres_new = centres_new';
load('Train.mat');
load('GroundTruth.mat');
load('Test.mat');
load('TrueClass.mat');

w = rand(10,K)./2  - 0.25;
g = 0;
for epoch = 1:1
    for m = 1:length(Train)
        x = Train(m,:);
        d = GroundTruth;
        
        % Gaussian RBF Kernel, or RBF Kernel: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
        % Applying Kernel on the Test Data
        for i = 1:K
            g(i,m) = exp(-(x-centres_new(:,i)')*(x-centres_new(:,i)')'/(2*spread(i)^2));
        end
    end
    
    % Implementation of the Least Squares algorithm
    w=pinv(g)'*d;
end

    % Applying Kernel on the Test Data
    for m = 1:length(Test)
        x = Test(m,:);

        for i = 1:K
            g_test(i,m) = exp(-(x-centres_new(:,i)')*(x-centres_new(:,i)')'/(2*spread(i)^2));
        end
    end

    % Predicting the values and finding Classification Rate.
    PREDICTIONS = (g_test'*w);
    for z = 1:length(Test)
        [val, col] = max(PREDICTIONS(z,:));
        predict(z,:) = col-1;
    end
    CR = (sum(predict==TrueClass)/length(TrueClass))*100
    
    % Visualizations
    figure;
    plot(1:epoch,CR,'b--o');
    title('Recognition Curve');
    xlabel('Number of epochs');
    ylabel('Classification Rate (%)');
