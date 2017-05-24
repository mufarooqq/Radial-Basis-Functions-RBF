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
%centres_new = centres_new';
load('Train.mat');
load('GroundTruth.mat');
load('Test.mat');
load('TrueClass.mat');

TD = Train';
GT = GroundTruth';
w = rand(10,K)./2  - 0.25;
eta = 2E-5;
g = 0;

for epoch = 1:100
    shuffle = randperm(60000);
    TD = TD(:,shuffle);
    GT = GT(:,shuffle);
    
    for m = 1:length(Train)
        x = TD(1:785,m);
        d = GT(1:10,m);

        % Gaussian RBF Kernel, or RBF Kernel: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
        % Applying Kernel on the Test Data        
        for i = 1:K
            g(i,m) = exp(-(x-centres_new(i,:)')'*(x-centres_new(i,:)')/(2*spread(i)^2));
        end
        
        % Implementation of the Least Mean Squares algorithm
        e = d - w*g(:,m);
        w_delta = eta.*e*g(:,m)';
        ee(:,m) = e;
        w = w + w_delta;
    end
    em = mean(ee.^2);
    mse(epoch) = em(1);
    fprintf('  Epoch # %d, MSE: %4.2f\n',epoch,mse(epoch));
    
    % Stopping Criteria
    if mse(epoch) < 1E-9
        break;
    end
    
    % Applying Kernel on the Test Data
    for m = 1:length(Test)
    x = Test(m,:);    
    for i = 1:K
        g_test(i,m) = exp(-(x-centres_new(i,:))*(x-centres_new(i,:))'/(2*spread(i)^2));
    end
    end

% Predicting the values and finding Classification Rate.
PREDICTIONS = (w*g_test)';
for z = 1:length(Test)
    [val, col] = max(PREDICTIONS(z,:));
    predict(z,:) = col-1;
end
CR(epoch) = (sum(predict==TrueClass)/length(TrueClass))*100

end

% Visualizations
figure;
plot(1:epoch,CR,'b--o');
title('Recognition Curve');
xlabel('Number of epochs');
ylabel('Classification Rate (%)');
