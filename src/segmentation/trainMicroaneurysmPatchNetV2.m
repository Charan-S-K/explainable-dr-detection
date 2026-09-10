%% SeeBeyond - Microaneurysm Patch U-Net V2
% V2 adds vessel hard-negative training examples.
% V1 model is preserved.

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

fprintf('\n==============================================\n');
fprintf('MICROANEURYSM PATCH U-NET V2 TRAINING\n');
fprintf('==============================================\n');

%% ------------------------------------------------
% Load split
%% ------------------------------------------------

load('models/microaneurysmV2Split.mat', ...
    'trainMAIdx','valMAIdx');

%% ------------------------------------------------
% V2 patch dataset
%% ------------------------------------------------

patchRoot = fullfile(pwd, ...
    'datasets','IDRiD','patches','microaneurysmsV2');

patchImageDir = fullfile(patchRoot,'images');
patchMaskDir  = fullfile(patchRoot,'masks');

if ~isfolder(patchImageDir)
    error('V2 patch image folder not found.');
end

if ~isfolder(patchMaskDir)
    error('V2 patch mask folder not found.');
end

imdsMA = imageDatastore(patchImageDir);

classNamesMA = ["background","microaneurysm"];

pxdsMA = pixelLabelDatastore( ...
    patchMaskDir, ...
    classNamesMA, ...
    [0 1], ...
    FileExtensions=".png");

fprintf('\nPatch images : %d\n', ...
    numel(imdsMA.Files));

fprintf('Patch masks  : %d\n', ...
    numel(pxdsMA.Files));

%% ------------------------------------------------
% Class distribution
%% ------------------------------------------------

countsMA = countEachLabel(pxdsMA);

disp(countsMA);

freqMA = countsMA.PixelCount ./ ...
    sum(countsMA.PixelCount);

fprintf('\nBackground frequency    : %.8f\n', ...
    freqMA(1));

fprintf('Microaneurysm frequency : %.8f\n', ...
    freqMA(2));

%% ------------------------------------------------
% Class weighting
%
% Limit the MA weight to avoid unstable training.
%% ------------------------------------------------

suggestedWeightMA = ...
    sqrt(freqMA(1) / max(freqMA(2),eps));

maWeight = min(2.5,suggestedWeightMA);

weightsMA = [1; maWeight];

fprintf('\nBackground weight    : %.3f\n', ...
    weightsMA(1));

fprintf('Microaneurysm weight : %.3f\n', ...
    weightsMA(2));

classWeightsMA = ...
    dlarray(weightsMA,"C");

%% ------------------------------------------------
% Combine dataset
%% ------------------------------------------------

dsMA = combine( ...
    imdsMA, ...
    pxdsMA);

%% ------------------------------------------------
% Safe augmentation
%% ------------------------------------------------

dsMAAug = transform( ...
    dsMA, ...
    @augmentLesionData);

sampleMA = preview(dsMAAug);

fprintf('\nSample image size:\n');
disp(size(sampleMA{1}));

fprintf('Sample mask size:\n');
disp(size(sampleMA{2}));

fprintf('Undefined pixels: %d\n', ...
    sum(isundefined(sampleMA{2}(:))));

%% ------------------------------------------------
% Network
%% ------------------------------------------------

fprintf('\nCreating U-Net V2...\n');

netMA = unet( ...
    [256 256 3], ...
    2, ...
    EncoderDepth=3, ...
    NumFirstEncoderFilters=16);

%% ------------------------------------------------
% Loss
%% ------------------------------------------------

lossMA = @(Y,T) crossentropy( ...
    Y, ...
    T, ...
    classWeightsMA, ...
    NormalizationFactor="all-elements");

%% ------------------------------------------------
% Training options
%% ------------------------------------------------

optionsMA = trainingOptions("adam", ...
    InitialLearnRate=5e-5, ...
    MaxEpochs=30, ...
    MiniBatchSize=4, ...
    Shuffle="every-epoch", ...
    ExecutionEnvironment="gpu", ...
    Plots="training-progress", ...
    Verbose=true);

%% ------------------------------------------------
% Train
%% ------------------------------------------------

fprintf('\nStarting V2 training...\n');

microaneurysmPatchNetV2 = trainnet( ...
    dsMAAug, ...
    netMA, ...
    lossMA, ...
    optionsMA);

%% ------------------------------------------------
% Save
%% ------------------------------------------------

save( ...
    'models/microaneurysmPatchNetV2.mat', ...
    'microaneurysmPatchNetV2', ...
    'weightsMA', ...
    'freqMA', ...
    'trainMAIdx', ...
    'valMAIdx');

fprintf('\n==============================================\n');
fprintf('MICROANEURYSM V2 TRAINING COMPLETE\n');
fprintf('==============================================\n');

fprintf('Saved:\n');
fprintf('models/microaneurysmPatchNetV2.mat\n');

fprintf('==============================================\n');