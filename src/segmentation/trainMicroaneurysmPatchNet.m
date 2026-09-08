%% SeeBeyond - Microaneurysm Patch U-Net
% Original-resolution 256x256 patch model

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

fprintf('\n===== MICROANEURYSM PATCH U-NET =====\n');

%% Load split

load('models/microaneurysmSplit.mat', ...
    'trainMAIdx','valMAIdx');

%% Paths

patchRoot = fullfile(pwd, ...
    'datasets','IDRiD','patches','microaneurysms');

patchImageDir = fullfile(patchRoot,'images');
patchMaskDir  = fullfile(patchRoot,'masks');

%% Datastores

imdsMAPatch = imageDatastore(patchImageDir);

classNamesMA = ["background","microaneurysm"];

pxdsMAPatch = pixelLabelDatastore( ...
    patchMaskDir, ...
    classNamesMA, ...
    [0 1], ...
    FileExtensions=".png");

fprintf('Patch images : %d\n',numel(imdsMAPatch.Files));
fprintf('Patch masks  : %d\n',numel(pxdsMAPatch.Files));

%% Class statistics

countsMA = countEachLabel(pxdsMAPatch);
disp(countsMA)

freqMA = ...
    countsMA.PixelCount ./ sum(countsMA.PixelCount);

fprintf('\nBackground frequency    : %.6f\n',freqMA(1));
fprintf('Microaneurysm frequency : %.6f\n',freqMA(2));

%% Mild class weighting
% MA pixels are extremely rare, but excessive weighting
% would create many false positives.

suggestedWeightMA = ...
    sqrt(freqMA(1) / max(freqMA(2),eps));

maWeight = min(2.0,suggestedWeightMA);

weightsMA = [1; maWeight];

fprintf('\nBackground weight    : %.3f\n',weightsMA(1));
fprintf('Microaneurysm weight : %.3f\n',weightsMA(2));

classWeightsMA = dlarray(weightsMA,"C");

%% Combine data

dsMA = combine(imdsMAPatch,pxdsMAPatch);

%% Safe augmentation

dsMAAug = transform(dsMA,@augmentLesionData);

%% Verify sample

sampleMA = preview(dsMAAug);

fprintf('\nImage size:\n');
disp(size(sampleMA{1}));

fprintf('Mask size:\n');
disp(size(sampleMA{2}));

fprintf('Undefined pixels: %d\n', ...
    sum(isundefined(sampleMA{2}(:))));

%% Lightweight U-Net

netMA = unet( ...
    [256 256 3], ...
    2, ...
    EncoderDepth=3, ...
    NumFirstEncoderFilters=16);

%% Weighted loss

lossMA = @(Y,T) crossentropy( ...
    Y,T, ...
    classWeightsMA, ...
    NormalizationFactor="all-elements");

%% Training options

optionsMA = trainingOptions("adam", ...
    InitialLearnRate=1e-4, ...
    MaxEpochs=20, ...
    MiniBatchSize=4, ...
    Shuffle="every-epoch", ...
    ExecutionEnvironment="gpu", ...
    Plots="training-progress", ...
    Verbose=true);

%% Train

microaneurysmPatchNet = trainnet( ...
    dsMAAug, ...
    netMA, ...
    lossMA, ...
    optionsMA);

%% Save immediately

save('models/microaneurysmPatchNet.mat', ...
    'microaneurysmPatchNet', ...
    'weightsMA', ...
    'freqMA', ...
    'trainMAIdx', ...
    'valMAIdx');

fprintf('\n========================================\n');
fprintf('MICROANEURYSM PATCH TRAINING COMPLETE\n');
fprintf('Saved: models/microaneurysmPatchNet.mat\n');
fprintf('========================================\n');