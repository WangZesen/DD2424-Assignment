% Initialization
clear all
clc
rng(400);

disp("Task 1: Calculate Ralative Error");
disp("Task 2: Show Costs and Visualized W for Different Setting of Parameters");
disp("Task 3 (Optional): Optimize the Network with All Three Methods");
disp("Task 4 (Optional): Optimize the Network with Larger Train Set");
disp("Task 5 (Optional): Optimize the Network with Learning Rate Decay");
disp("Task 6 (Optional): Optimize the Network with Xavier Initialization");
disp("Task 7 (Optional): Train with SVM Multi-class Loss");
task_label = input('Task #: ', 's');

% Set model parameters
K = 10;
eps = 1e-12;
lambda = 0.1;
[tmp, ~, ~] = LoadBatch('data_batch_1.mat');
d = size(tmp, 1);
W = normrnd(0, 0.01, [K, d]);
b = normrnd(0, 0.01, [K, 1]);

if ("1" <= task_label) && (task_label <= "2")
    % Load data
    [trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
    [valX, valY, valy] = LoadBatch('data_batch_2.mat');
    [testX, testY, testy] = LoadBatch('test_batch.mat');
    if task_label == "1"
        lambda = 0.1;
        % Evaluate classifier
        P = EvaluateClassifier(trainX(:, 1:100), W, b);
        % Calculate Gradient
        [grad_W, grad_b] = ComputeGradients(trainX(:, 1:100), trainY(:, 1:100), P, W, lambda);
        [grad_b1, grad_W1] = ComputeGradsNumSlow(trainX(:, 1:100), trainY(:, 1:100), W, b, lambda, 1e-6);
        % Calculate relative error of gradients for W and b
        absolute_error_W = abs(grad_W - grad_W1);
        error_index_W = max(eps, abs(grad_W) + abs(grad_W1));
        mean_relative_error_W = sum(sum(absolute_error_W ./ error_index_W)) / numel(grad_W);
        disp("mean relative error of W = " + mean_relative_error_W);
        absolute_error_b = abs(grad_b - grad_b1);
        error_index_b = max(eps, abs(grad_b) + abs(grad_b1));
        mean_relative_error_b = sum(sum(absolute_error_b ./ error_index_b)) / numel(grad_b);
        disp("mean relative error of b = " + mean_relative_error_b);
    else
        
        n_batch = [100, 100, 100, 100];
        n_epoch = [40, 40, 40, 40];
        eta = [0.1, 0.01, 0.01, 0.01];
        lambda = [0, 0, 0.1, 1.0];
%         n_batch = [100];
%         n_epoch = [40];
%         eta = [0.01];
%         lambda = [0.1];
        
        trainCost = zeros(40, length(n_batch));
        valCost = zeros(40, length(n_batch));
        
        for k = 1 : length(n_batch)
            testParam = containers.Map;
            testParam('n_batch') = n_batch(k);
            testParam('eta') = eta(k);
            testParam('n_epochs') = n_epoch(k);
            disp("Training with n_batch=" + n_batch(k) + ", eta=" + eta(k) + ", n_epochs=" + n_epoch(k) + ", lambda=" + lambda(k));
            [trainCost(:, k), valCost(:, k), Wstar, bstar] = MiniBatchGD(trainX, trainY, valX, valY, testParam, W, b, lambda(k));
            disp("Accuracy on Test set = " + ComputeAccuracy(testX, testy, Wstar, bstar));
            clear s_im
            for i = 1 : 10
                im = reshape(Wstar(i, :), 32, 32, 3);
                s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
                s_im{i} = permute(s_im{i}, [2, 1, 3]);
                subplot(2, 5, i), imshow(s_im{i}, 'Border', 'tight')
            end
            disp("Press any key to move on...")
            pause;
            clf
            hold on
            plot(1 : n_epoch(k), trainCost(:, k));
            plot(1 : n_epoch(k), valCost(:, k));
            legend('train', 'val');
            disp("Press any key to move on...")
            pause;
            clf
        end
    end
end
if task_label == "3"
    [trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
    % Concatenate all training data
    for i = 2 : 5
        [tmpX, tmpY, tmpy] = LoadBatch("data_batch_" + i + ".mat");
        trainX = cat(2, trainX, tmpX);
        trainY = cat(2, trainY, tmpY);
        trainy = cat(2, trainy, tmpy);
    end
    % Fetch validation set
    valX = trainX(:, 1:1000);
    valY = trainY(:, 1:1000);
    valy = trainy(1:1000);
    trainX = trainX(:, 1001:size(trainX, 2));
    trainY = trainY(:, 1001:size(trainY, 2));
    trainy = trainy(1001:size(trainy, 2));
    [testX, testY, testy] = LoadBatch('test_batch.mat');
    
    % Xavier initialization
    W = normrnd(0, 2 / (K + d), [K, d]);
    b = normrnd(0, 2 / (K + d), [K, d]);
    
    testParam = containers.Map;
    testParam('n_batch') = 100;
    testParam('eta') = 0.01;
    testParam('n_epochs') = 40;
    lambda = 0;
    
    disp("Training with n_batch=" + testParam('n_batch') + ", eta=" + testParam('eta') + ", n_epochs=" + testParam('n_epochs') + ", lambda=" + lambda);
    [trainCost, valCost, Wstar, bstar] = MiniBatchGD_Optim(trainX, trainY, valX, valY, testParam, W, b, lambda, 0.1);
    disp("Accuracy on Test set = " + ComputeAccuracy(testX, testy, Wstar, bstar));
    clear s_im
    for i = 1 : 10
        im = reshape(Wstar(i, :), 32, 32, 3);
        s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
        s_im{i} = permute(s_im{i}, [2, 1, 3]);
        subplot(2, 5, i), imshow(s_im{i}, 'Border', 'tight')
    end
    disp("Press any key to move on...")
    pause;
    clf
    hold on
    plot(1 : testParam('n_epochs'), trainCost);
    plot(1 : testParam('n_epochs'), valCost);
    legend('train', 'val');
    disp("Press any key to move on...");
    pause;
    clf
end

if task_label == "4"
    [trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
    % Concatenate all training data
    for i = 2 : 5
        [tmpX, tmpY, tmpy] = LoadBatch("data_batch_" + i + ".mat");
        trainX = cat(2, trainX, tmpX);
        trainY = cat(2, trainY, tmpY);
        trainy = cat(2, trainy, tmpy);
    end
    % Fetch validation set
    valX = trainX(:, 1:1000);
    valY = trainY(:, 1:1000);
    valy = trainy(1:1000);
    trainX = trainX(:, 1001:size(trainX, 2));
    trainY = trainY(:, 1001:size(trainY, 2));
    trainy = trainy(1001:size(trainy, 2));
    [testX, testY, testy] = LoadBatch('test_batch.mat');
    
    W = normrnd(0, 0.01, [K, d]);
    b = normrnd(0, 0.01, [K, d]);
    
    testParam = containers.Map;
    testParam('n_batch') = 100;
    testParam('eta') = 0.01;
    testParam('n_epochs') = 40;
    lambda = 0;
    
    disp("Training with n_batch=" + testParam('n_batch') + ", eta=" + testParam('eta') + ", n_epochs=" + testParam('n_epochs') + ", lambda=" + lambda);
    [trainCost, valCost, Wstar, bstar] = MiniBatchGD(trainX, trainY, valX, valY, testParam, W, b, lambda);
    disp("Accuracy on Test set = " + ComputeAccuracy(testX, testy, Wstar, bstar));
    clear s_im
    for i = 1 : 10
        im = reshape(Wstar(i, :), 32, 32, 3);
        s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
        s_im{i} = permute(s_im{i}, [2, 1, 3]);
        subplot(2, 5, i), imshow(s_im{i}, 'Border', 'tight')
    end
    disp("Press any key to move on...")
    pause;
    clf
    hold on
    plot(1 : testParam('n_epochs'), trainCost);
    plot(1 : testParam('n_epochs'), valCost);
    legend('train', 'val');
    disp("Press any key to move on...");
    pause;
    clf
end

if task_label == "5"
    [trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
    [valX, valY, valy] = LoadBatch('data_batch_2.mat');
    [testX, testY, testy] = LoadBatch('test_batch.mat');
    
    W = normrnd(0, 0.01, [K, d]);
    b = normrnd(0, 0.01, [K, d]);
    
    testParam = containers.Map;
    testParam('n_batch') = 100;
    testParam('eta') = 0.01;
    testParam('n_epochs') = 40;
    lambda = 0;
    
    disp("Training with n_batch=" + testParam('n_batch') + ", eta=" + testParam('eta') + ", n_epochs=" + testParam('n_epochs') + ", lambda=" + lambda);
    [trainCost, valCost, Wstar, bstar] = MiniBatchGD_Optim(trainX, trainY, valX, valY, testParam, W, b, lambda, 0.1);
    disp("Accuracy on Test set = " + ComputeAccuracy(testX, testy, Wstar, bstar));
    clear s_im
    for i = 1 : 10
        im = reshape(Wstar(i, :), 32, 32, 3);
        s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
        s_im{i} = permute(s_im{i}, [2, 1, 3]);
        subplot(2, 5, i), imshow(s_im{i}, 'Border', 'tight')
    end
    disp("Press any key to move on...")
    pause;
    clf
    hold on
    plot(1 : testParam('n_epochs'), trainCost);
    plot(1 : testParam('n_epochs'), valCost);
    legend('train', 'val');
    disp("Press any key to move on...");
    pause;
    clf
end

if task_label == "6"
    [trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
    [valX, valY, valy] = LoadBatch('data_batch_2.mat');
    [testX, testY, testy] = LoadBatch('test_batch.mat');
    
    % Xavier Initialization
    W = normrnd(0, 2 / (K + d), [K, d]);
    b = normrnd(0, 2 / (K + d), [K, d]);
    
    testParam = containers.Map;
    testParam('n_batch') = 100;
    testParam('eta') = 0.01;
    testParam('n_epochs') = 40;
    lambda = 0;
    
    disp("Training with n_batch=" + testParam('n_batch') + ", eta=" + testParam('eta') + ", n_epochs=" + testParam('n_epochs') + ", lambda=" + lambda);
    [trainCost, valCost, Wstar, bstar] = MiniBatchGD(trainX, trainY, valX, valY, testParam, W, b, lambda);
    disp("Accuracy on Test set = " + ComputeAccuracy(testX, testy, Wstar, bstar));
    clear s_im
    for i = 1 : 10
        im = reshape(Wstar(i, :), 32, 32, 3);
        s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
        s_im{i} = permute(s_im{i}, [2, 1, 3]);
        subplot(2, 5, i), imshow(s_im{i}, 'Border', 'tight')
    end
    disp("Press any key to move on...")
    pause;
    clf
    hold on
    plot(1 : testParam('n_epochs'), trainCost);
    plot(1 : testParam('n_epochs'), valCost);
    legend('train', 'val');
    disp("Press any key to move on...");
    pause;
    clf
end

if task_label == "7"
    clf;
    [trainX, trainY, trainy] = LoadBatch_SVM('data_batch_1.mat');
    [valX, valY, valy] = LoadBatch_SVM('data_batch_2.mat');
    [testX, testY, testy] = LoadBatch_SVM('test_batch.mat');
    lambda = 1e-3;
    
    testParam = containers.Map;
    testParam('n_batch') = 100;
    testParam('eta') = 0.001;
    testParam('n_epochs') = 60;
    
    [trainCost, valCost, Wstar, bstar] = MiniBatchGD_SVM(trainX, trainy, valX, valy, testParam, W, b, lambda);
    disp("Test Accuracy: " + ComputeAccuracy_SVM(testX, testy, Wstar, bstar));
    hold on
    plot(1 : testParam('n_epochs'), trainCost);
    plot(1 : testParam('n_epochs'), valCost);
    legend('train', 'val');
    disp("Press any key to move on...");
    pause;
    clf;
end










% Functions are defined below
function [X, Y, y] = LoadBatch(filename)
    addpath ./Datasets/cifar-10-batches-mat/
    A = load(filename);
    X = double(A.data).' / 255.0;
    y = double(A.labels);
    n = size(X, 2);
    Y = zeros(10, n, 'double');
    for i = 1 : n
        Y(y(i) + 1, i) = 1;
    end
end

function [X, Y, y] = LoadBatch_SVM(filename)
    addpath ./Datasets/cifar-10-batches-mat/
    A = load(filename);
    X = double(A.data).' / 255.0;
    y = double(A.labels);
    n = size(X, 2);
    Y = repmat(-1, 10, n);
    for i = 1 : n
        Y(y(i) + 1, i) = 1;
    end
end

function acc = ComputeAccuracy(X, y, W, b)
    P = EvaluateClassifier(X, W, b);
    n = size(P, 2);
    acc = 0;
    for i = 1 : n
        [~, argmax] = max(P(:, i));
        if argmax == y(i) + 1
            acc = acc + 1;
        end
    end
    acc = acc / n;
end

function acc = ComputeAccuracy_SVM(X, y, W, b)
    P = EvaluateClassifier_SVM(X, W, b);
    n = size(P, 2);
    acc = 0;
    for i = 1 : n
        [~, argmax] = max(P(:, i));
        if argmax == y(i) + 1
            acc = acc + 1;
        end
    end
    acc = acc / n;
end


function P = EvaluateClassifier(X, W, b)
    P = W * X;
    for i = 1 : 10
        P(i, :) = P(i, :) + b(i, 1);
    end
    P = exp(P);
    n = size(P, 2);
    for i = 1 : n
        P(:, i) = P(:, i) / sum(P(:, i));
    end
end

function P = EvaluateClassifier_SVM(X, W, b)
    P = W * X;
    for i = 1 : 10
        P(i, :) = P(i, :) + b(i, 1);
    end
end

function J = ComputeCost(X, Y, W, b, lambda)
    P = EvaluateClassifier(X, W, b);
    P = log(P);
    J = - sum(sum(P .* Y)) / size(X, 2) + sum(sum(W .* W)) * lambda;
end

function J = ComputeCost_SVM(X, y, W, b, lambda)
    P = EvaluateClassifier_SVM(X, W, b);
    n = size(X, 2);
    J = 0;
    for i = 1 : n
        P(:, i) = P(:, i) - P(y(i) + 1, i) + 1;
        P(:, i) = max(P(:, i), 0);
        J = J + sum(P(:, i)) - 1;
    end
    J = J / n;
    J = J + lambda * sum(sum(W .* W));
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
    n = size(X, 2);
    d = size(X, 1);
    K = 10;
    grad_W = zeros(K, d);
    grad_b = zeros(K, 1);
    for i = 1 : n
        grad_W = grad_W + (P(:, i) - Y(:, i)) * X(:, i).';
        grad_b = grad_b + P(:, i) - Y(:, i);
    end
    grad_W = grad_W / n + 2 * lambda * W;
    grad_b = grad_b / n;
end

function [grad_W, grad_b] = ComputeGradients_SVM(X, y, P, W, lambda)
    n = size(X, 2);
    d = size(X, 1);
    K = 10;
    grad_W = zeros(K, d);
    grad_b = zeros(K, 1);
    for i = 1 : n
        P(:, i) = P(:, i) - P(y(i) + 1, i) + 1;
        P(:, i) = max(P(:, i), 0) > 0;
        P(y(i) + 1, i) = P(y(i) + 1, i) - sum(P(:, i));
        grad_W = grad_W + P(:, i) * X(:, i).';
        grad_b = grad_b + P(:, i);
        % grad_W(y(i) + 1, :) = grad_W(y(i) + 1, :) - sum(P(:, i)) * X(:, i).';
    end
    grad_W = grad_W / n + 2 * lambda * W;
    grad_b = grad_b / n;
end

function [trainCost, valCost, Wstar, bstar] = MiniBatchGD(X, Y, valX, valY, params, W, b, lambda)
    trainCost = zeros(params('n_epochs'), 1);
    valCost = zeros(params('n_epochs'), 1);
    Wstar = repmat(W, 1);
    bstar = repmat(b, 1);
    N = size(X, 2);
    for i = 1 : params('n_epochs')
        for j = 1 : N / params('n_batch')
            j_start = (j - 1) * params('n_batch') + 1;
            j_end = j * params('n_batch');
            Xbatch = X(:, j_start:j_end);
            Ybatch = Y(:, j_start:j_end);
            P = EvaluateClassifier(Xbatch, Wstar, bstar);
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, Wstar, lambda);
            
            
            Wstar = Wstar - grad_W * params('eta');
            bstar = bstar - grad_b * params('eta');
        end
        valCost(i) = ComputeCost(valX, valY, Wstar, bstar, lambda);
        trainCost(i) = ComputeCost(X, Y, Wstar, bstar, lambda);
    end
end

function [trainCost, valCost, Wstar, bstar] = MiniBatchGD_Optim(X, Y, valX, valY, params, W, b, lambda, k)
    trainCost = zeros(params('n_epochs'), 1);
    valCost = zeros(params('n_epochs'), 1);
    Wstar = repmat(W, 1);
    bstar = repmat(b, 1);
    N = size(X, 2);
    for i = 1 : params('n_epochs')
        for j = 1 : N / params('n_batch')
            j_start = (j - 1) * params('n_batch') + 1;
            j_end = j * params('n_batch');
            inds = j_start : j_end;
            Xbatch = X(:, inds);
            Ybatch = Y(:, inds);
            P = EvaluateClassifier(Xbatch, Wstar, bstar);
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, Wstar, lambda);
            Wstar = Wstar - grad_W * params('eta') * exp(- k * (i - 1)); % lr decay
            bstar = bstar - grad_b * params('eta') * exp(- k * (i - 1)); % lr decay
        end
        valCost(i) = ComputeCost(valX, valY, Wstar, bstar, lambda);
        trainCost(i) = ComputeCost(X, Y, Wstar, bstar, lambda);
        
    end
end

function [trainCost, valCost, Wstar, bstar] = MiniBatchGD_SVM(X, Y, valX, valY, params, W, b, lambda)
    trainCost = zeros(params('n_epochs'), 1);
    valCost = zeros(params('n_epochs'), 1);
    Wstar = repmat(W, 1);
    bstar = repmat(b, 1);
    N = size(X, 2);
    for i = 1 : params('n_epochs')
        for j = 1 : N / params('n_batch')
            j_start = (j - 1) * params('n_batch') + 1;
            j_end = j * params('n_batch');
            inds = j_start : j_end;
            Xbatch = X(:, inds);
            Ybatch = Y(inds);
            P = EvaluateClassifier_SVM(Xbatch, Wstar, bstar);
            [grad_W, grad_b] = ComputeGradients_SVM(Xbatch, Ybatch, P, Wstar, lambda);
            Wstar = Wstar - grad_W * params('eta');
            bstar = bstar - grad_b * params('eta');
        end
        valCost(i) = ComputeCost_SVM(valX, valY, Wstar, bstar, lambda);
        trainCost(i) = ComputeCost_SVM(X, Y, Wstar, bstar, lambda);
    end
end

function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)
    no = size(W, 1);
    grad_W = zeros(size(W));
    grad_b = zeros(no, 1);
    for i=1:length(b)
        b_try = b;
        b_try(i) = b_try(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        b_try = b;
        b_try(i) = b_try(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b(i) = (c2-c1) / (2*h);
    end
    for i=1:numel(W)
        W_try = W;
        W_try(i) = W_try(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);
        W_try = W;
        W_try(i) = W_try(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
        grad_W(i) = (c2-c1) / (2*h);
    end
end

function [grad_b, grad_W] = ComputeGradsNumSlow_SVM(X, y, W, b, lambda, h)
    no = size(W, 1);
    grad_W = zeros(size(W));
    grad_b = zeros(no, 1);
    for i=1:length(b)
        b_try = b;
        b_try(i) = b_try(i) - h;
        c1 = ComputeCost_SVM(X, y, W, b_try, lambda);
        b_try = b;
        b_try(i) = b_try(i) + h;
        c2 = ComputeCost_SVM(X, y, W, b_try, lambda);
        grad_b(i) = (c2-c1) / (2*h);
    end
    for i=1:numel(W)
        W_try = W;
        W_try(i) = W_try(i) - h;
        c1 = ComputeCost_SVM(X, y, W_try, b, lambda);
        W_try = W;
        W_try(i) = W_try(i) + h;
        c2 = ComputeCost_SVM(X, y, W_try, b, lambda);
        grad_W(i) = (c2-c1) / (2*h);
    end
end