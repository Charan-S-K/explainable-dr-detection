%% SeeBeyond - APTOS 2019 Stratified Dataset Split
% Stage 3: Diabetic Retinopathy Severity Classification

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

rng(42);

fprintf('\n========================================\n');
fprintf('APTOS 2019 STRATIFIED DATASET SPLIT\n');
fprintf('========================================\n');

%% Dataset paths

aptosCSV = fullfile(pwd, ...
    'datasets','APTOS2019','train.csv');

aptosImageDir = fullfile(pwd, ...
    'datasets','APTOS2019','train_images');

%% Read labels

aptosTable = readtable(aptosCSV);

N = height(aptosTable);

fprintf('Total labelled images : %d\n',N);

%% Construct full image paths

imagePaths = fullfile( ...
    aptosImageDir, ...
    string(aptosTable.id_code) + ".png");

%% Verify every image exists

existsMask = isfile(imagePaths);

fprintf('Images found          : %d\n',sum(existsMask));
fprintf('Missing images        : %d\n',sum(~existsMask));

if any(~existsMask)

    error('Some APTOS images are missing.');

end

%% Classification labels

classNames = [
    "No_DR"
    "Mild"
    "Moderate"
    "Severe"
    "Proliferative_DR"
];

labels = categorical( ...
    aptosTable.diagnosis, ...
    0:4, ...
    classNames);

%% =========================================================
% Stratified 70 / 15 / 15 split
%% =========================================================

trainIdx = [];
valIdx   = [];
testIdx  = [];

fprintf('\n===== CLASS-WISE SPLIT =====\n');

for grade = 0:4

    idx = find(aptosTable.diagnosis == grade);

    % Randomize only inside this class
    idx = idx(randperm(numel(idx)));

    n = numel(idx);

    nTrain = floor(0.70*n);
    nVal   = floor(0.15*n);

    nTest = n - nTrain - nVal;

    trainPart = idx(1:nTrain);

    valPart = idx( ...
        nTrain+1 : nTrain+nVal);

    testPart = idx( ...
        nTrain+nVal+1 : end);

    trainIdx = [trainIdx; trainPart(:)];
    valIdx   = [valIdx; valPart(:)];
    testIdx  = [testIdx; testPart(:)];

    fprintf(['Grade %d : Total=%4d | ' ...
        'Train=%4d | Val=%3d | Test=%3d\n'], ...
        grade,n,nTrain,nVal,nTest);

end

%% Shuffle each final split

trainIdx = trainIdx(randperm(numel(trainIdx)));
valIdx   = valIdx(randperm(numel(valIdx)));
testIdx  = testIdx(randperm(numel(testIdx)));

%% =========================================================
% Safety checks
%% =========================================================

assert(isempty(intersect(trainIdx,valIdx)));
assert(isempty(intersect(trainIdx,testIdx)));
assert(isempty(intersect(valIdx,testIdx)));

allIndices = [
    trainIdx
    valIdx
    testIdx
];

assert(numel(unique(allIndices)) == N);

fprintf('\nNo overlap between splits : YES\n');
fprintf('All images accounted for : YES\n');

%% =========================================================
% Create split tables
%% =========================================================

trainTable = table( ...
    imagePaths(trainIdx), ...
    labels(trainIdx), ...
    aptosTable.diagnosis(trainIdx), ...
    'VariableNames', ...
    {'File','Label','Grade'});

valTable = table( ...
    imagePaths(valIdx), ...
    labels(valIdx), ...
    aptosTable.diagnosis(valIdx), ...
    'VariableNames', ...
    {'File','Label','Grade'});

testTable = table( ...
    imagePaths(testIdx), ...
    labels(testIdx), ...
    aptosTable.diagnosis(testIdx), ...
    'VariableNames', ...
    {'File','Label','Grade'});

%% =========================================================
% Create image datastores
%% =========================================================

imdsTrainAPTOS = imageDatastore( ...
    trainTable.File, ...
    Labels=trainTable.Label);

imdsValAPTOS = imageDatastore( ...
    valTable.File, ...
    Labels=valTable.Label);

imdsTestAPTOS = imageDatastore( ...
    testTable.File, ...
    Labels=testTable.Label);

%% Overall counts

fprintf('\n========================================\n');
fprintf('FINAL SPLIT COUNTS\n');
fprintf('========================================\n');

fprintf('Training   : %d\n',numel(trainIdx));
fprintf('Validation : %d\n',numel(valIdx));
fprintf('Testing    : %d\n',numel(testIdx));

fprintf('Total      : %d\n', ...
    numel(trainIdx)+numel(valIdx)+numel(testIdx));

%% Show class distribution in each split

fprintf('\n===== TRAINING CLASSES =====\n');
disp(countEachLabel(imdsTrainAPTOS))

fprintf('\n===== VALIDATION CLASSES =====\n');
disp(countEachLabel(imdsValAPTOS))

fprintf('\n===== TEST CLASSES =====\n');
disp(countEachLabel(imdsTestAPTOS))

%% Save split

save('models/aptosStage3Split.mat', ...
    'trainIdx', ...
    'valIdx', ...
    'testIdx', ...
    'trainTable', ...
    'valTable', ...
    'testTable', ...
    'classNames');

fprintf('\nSaved: models/aptosStage3Split.mat\n');

fprintf('\n========================================\n');
fprintf('APTOS SPLIT COMPLETE\n');
fprintf('========================================\n');
