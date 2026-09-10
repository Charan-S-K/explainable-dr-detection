%% SeeBeyond - Stage 3 DR Severity Classifier
% APTOS 2019
% 5-class diabetic retinopathy classification
%
% Classes:
% 0 - No DR
% 1 - Mild
% 2 - Moderate
% 3 - Severe
% 4 - Proliferative DR

clearvars;
clc;

%% =========================================================
% Project setup
%% =========================================================

projectRoot = '/home/tharunkumar-s-v/Documents/SeeBeyond';

cd(projectRoot);
addpath(genpath(fullfile(projectRoot,'src')));

fprintf('\n========================================\n');
fprintf('SEEBEYOND STAGE 3 DR CLASSIFIER\n');
fprintf('========================================\n');

%% =========================================================
% GPU
%% =========================================================

g = gpuDevice;

fprintf('\nGPU: %s\n',g.Name);
fprintf('GPU memory: %.2f GB\n', ...
    g.AvailableMemory/1e9);

%% =========================================================
% Load APTOS split
%% =========================================================

splitFile = fullfile( ...
    projectRoot, ...
    'models', ...
    'aptosStage3Split.mat');

S = load(splitFile);

trainTable = S.trainTable;
valTable   = S.valTable;
testTable  = S.testTable;
classNames = S.classNames;

fprintf('\nTraining images   : %d\n',height(trainTable));
fprintf('Validation images : %d\n',height(valTable));
fprintf('Testing images    : %d\n',height(testTable));

%% =========================================================
% Image datastores
%% =========================================================

imdsTrain = imageDatastore( ...
    trainTable.File, ...
    Labels=trainTable.Label);

imdsVal = imageDatastore( ...
    valTable.File, ...
    Labels=valTable.Label);

imdsTest = imageDatastore( ...
    testTable.File, ...
    Labels=testTable.Label);

%% =========================================================
% Class distribution
%% =========================================================

fprintf('\n===== TRAINING DISTRIBUTION =====\n');
disp(countEachLabel(imdsTrain));

fprintf('\n===== VALIDATION DISTRIBUTION =====\n');
disp(countEachLabel(imdsVal));

%% =========================================================
% Network
%% =========================================================

inputSize = [224 224 3];

numClasses = numel(classNames);

fprintf('\nNumber of classes : %d\n',numClasses);
fprintf('Input size        : %dx%dx%d\n',inputSize);

%% =========================================================
% Transfer learning network
%% =========================================================

fprintf('\nCreating pretrained ResNet-18...\n');

net = imagePretrainedNetwork( ...
    "resnet18", ...
    NumClasses=numClasses);

%% =========================================================
% Data augmentation
%% =========================================================

augmenter = imageDataAugmenter( ...
    RandXReflection=true, ...
    RandYReflection=true, ...
    RandRotation=[-10 10], ...
    RandXTranslation=[-8 8], ...
    RandYTranslation=[-8 8], ...
    RandXScale=[0.95 1.05], ...
    RandYScale=[0.95 1.05]);

augimdsTrain = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imdsTrain, ...
    DataAugmentation=augmenter, ...
    ColorPreprocessing="gray2rgb");

augimdsVal = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imdsVal, ...
    ColorPreprocessing="gray2rgb");

augimdsTest = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imdsTest, ...
    ColorPreprocessing="gray2rgb");

%% =========================================================
% Training options
%% =========================================================

miniBatchSize = 16;

options = trainingOptions("adam", ...
    InitialLearnRate=1e-4, ...
    MaxEpochs=15, ...
    MiniBatchSize=miniBatchSize, ...
    Shuffle="every-epoch", ...
    ValidationData=augimdsVal, ...
    ValidationFrequency=100, ...
    ValidationPatience=4, ...
    Verbose=true, ...
    Plots="training-progress", ...
    ExecutionEnvironment="gpu");

%% =========================================================
% Train classifier
%% =========================================================

fprintf('\n========================================\n');
fprintf('STARTING CLASSIFIER TRAINING\n');
fprintf('========================================\n');

tic;

[drClassifier,info] = trainnet( ...
    augimdsTrain, ...
    net, ...
    "crossentropy", ...
    options);

trainingTime = toc;

fprintf('\nTraining completed.\n');
fprintf('Training time: %.2f minutes\n', ...
    trainingTime/60);

%% =========================================================
% Validation prediction
%% =========================================================

fprintf('\nRunning validation prediction...\n');

valScores = minibatchpredict( ...
    drClassifier, ...
    augimdsVal, ...
    ExecutionEnvironment="gpu");

valScores = gather(valScores);

valPred = scores2label( ...
    valScores, ...
    classNames);

valTrue = imdsVal.Labels;

valAccuracy = mean(valPred == valTrue);

fprintf('\nValidation Accuracy: %.2f %%\n', ...
    valAccuracy*100);

%% =========================================================
% Test prediction
%% =========================================================

fprintf('\nRunning final TEST prediction...\n');

testScores = minibatchpredict( ...
    drClassifier, ...
    augimdsTest, ...
    ExecutionEnvironment="gpu");

testScores = gather(testScores);

testPred = scores2label( ...
    testScores, ...
    classNames);

testTrue = imdsTest.Labels;

%% =========================================================
% Test accuracy
%% =========================================================

testAccuracy = mean(testPred == testTrue);

fprintf('\n========================================\n');
fprintf('FINAL TEST RESULT\n');
fprintf('========================================\n');

fprintf('Test Accuracy: %.2f %%\n', ...
    testAccuracy*100);

%% =========================================================
% Confusion matrix
%% =========================================================

fprintf('\n===== CONFUSION MATRIX =====\n');

confMat = confusionmat( ...
    testTrue, ...
    testPred, ...
    'Order',categorical(classNames));

disp(confMat);

%% =========================================================
% Per-class precision / recall / F1
%% =========================================================

precision = zeros(numClasses,1);
recall    = zeros(numClasses,1);
f1        = zeros(numClasses,1);

for k = 1:numClasses

    TP = confMat(k,k);

    FP = sum(confMat(:,k)) - TP;

    FN = sum(confMat(k,:)) - TP;

    precision(k) = TP / max(TP + FP,1);

    recall(k) = TP / max(TP + FN,1);

    f1(k) = ...
        2 * precision(k) * recall(k) / ...
        max(precision(k) + recall(k),eps);

end

fprintf('\n===== PER-CLASS METRICS =====\n');

for k = 1:numClasses

    fprintf('\n%s\n',classNames(k));

    fprintf('Precision : %.4f\n', ...
        precision(k));

    fprintf('Recall    : %.4f\n', ...
        recall(k));

    fprintf('F1 Score  : %.4f\n', ...
        f1(k));

end

macroPrecision = mean(precision);
macroRecall    = mean(recall);
macroF1        = mean(f1);

fprintf('\n===== MACRO METRICS =====\n');

fprintf('Precision : %.4f\n',macroPrecision);
fprintf('Recall    : %.4f\n',macroRecall);
fprintf('F1 Score  : %.4f\n',macroF1);

%% =========================================================
% Save classifier
%% =========================================================

modelFile = fullfile( ...
    projectRoot, ...
    'models', ...
    'drClassifier.mat');

save(modelFile, ...
    'drClassifier', ...
    'classNames', ...
    'inputSize', ...
    'testAccuracy', ...
    'valAccuracy', ...
    'precision', ...
    'recall', ...
    'f1', ...
    'macroPrecision', ...
    'macroRecall', ...
    'macroF1', ...
    'confMat', ...
    'trainingTime', ...
    '-v7.3');

fprintf('\n========================================\n');
fprintf('MODEL SAVED\n');
fprintf('========================================\n');

fprintf('%s\n',modelFile);

fprintf('\n========================================\n');
fprintf('STAGE 3 CLASSIFIER COMPLETE\n');
fprintf('========================================\n');