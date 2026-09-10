%% SeeBeyond - Haemorrhage Patch U-Net V3
% Improved stable training using weighted cross-entropy.
%
% Uses existing haemorrhages_v2 patches.
% Does NOT overwrite old models.

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

rng(42);

fprintf('\n========================================\n');
fprintf('HAEMORRHAGE PATCH U-NET V3\n');
fprintf('========================================\n');

%% Load split

load('models/haemorrhageV2Split.mat', ...
    'trainHEIdx','valHEIdx', ...
    'positiveCount','vesselNegCount','normalNegCount');

%% Patch paths

patchRoot = fullfile(pwd, ...
    'datasets','IDRiD','patches','haemorrhages_v2');

patchImageDir = fullfile(patchRoot,'images');
patchMaskDir  = fullfile(patchRoot,'masks');

%% Datastores

imdsHEV3 = imageDatastore(patchImageDir);

classNamesHEV3 = [
    "background"
    "haemorrhage"
];

pxdsHEV3 = pixelLabelDatastore( ...
    patchMaskDir, ...
    classNamesHEV3, ...
    [0 1], ...
    FileExtensions=".png");

fprintf('Patch images : %d\n',numel(imdsHEV3.Files));
fprintf('Patch masks  : %d\n',numel(pxdsHEV3.Files));

%% Pixel statistics

countsHEV3 = countEachLabel(pxdsHEV3);
disp(countsHEV3)

freqHEV3 = ...
    countsHEV3.PixelCount ./ ...
    sum(countsHEV3.PixelCount);

fprintf('\nBackground frequency  : %.6f\n',freqHEV3(1));
fprintf('Haemorrhage frequency : %.6f\n',freqHEV3(2));

%% =========================================================
% Stable class weighting
%
% sqrt ratio prevents extremely aggressive weights.
%% =========================================================

suggestedWeight = ...
    sqrt(freqHEV3(1) / max(freqHEV3(2),eps));

haemorrhageWeightV3 = ...
    min(3.5,suggestedWeight);

weightsHEV3 = [
    1
    haemorrhageWeightV3
];

fprintf('\nBackground weight  : %.3f\n',weightsHEV3(1));
fprintf('Haemorrhage weight : %.3f\n',weightsHEV3(2));

classWeightsHEV3 = ...
    dlarray(weightsHEV3,"C");

%% Combine + augmentation

dsHEV3 = combine( ...
    imdsHEV3,pxdsHEV3);

dsHEV3Aug = transform( ...
    dsHEV3,@augmentLesionData);

%% Check sample

sample = preview(dsHEV3Aug);

fprintf('\nImage size:\n');
disp(size(sample{1}));

fprintf('Mask size:\n');
disp(size(sample{2}));

fprintf('Undefined pixels : %d\n', ...
    sum(isundefined(sample{2}(:))));

%% U-Net

netHEV3 = unet( ...
    [256 256 3], ...
    2, ...
    EncoderDepth=3, ...
    NumFirstEncoderFilters=16);

%% =========================================================
% Weighted cross entropy ONLY
%
% We deliberately remove Tversky here because V2 collapsed.
%% =========================================================

lossHEV3 = @(Y,T) crossentropy( ...
    Y,T, ...
    classWeightsHEV3, ...
    NormalizationFactor="all-elements");

%% Training options

optionsHEV3 = trainingOptions("adam", ...
    InitialLearnRate=1e-4, ...
    MaxEpochs=25, ...
    MiniBatchSize=4, ...
    Shuffle="every-epoch", ...
    ExecutionEnvironment="gpu", ...
    Plots="training-progress", ...
    Verbose=true);

%% Train

fprintf('\nStarting Haemorrhage V3 training...\n');

haemorrhagePatchNetV3 = trainnet( ...
    dsHEV3Aug, ...
    netHEV3, ...
    lossHEV3, ...
    optionsHEV3);

%% Save

save('models/haemorrhagePatchNetV3.mat', ...
    'haemorrhagePatchNetV3', ...
    'weightsHEV3', ...
    'freqHEV3', ...
    'trainHEIdx', ...
    'valHEIdx');

fprintf('\n========================================\n');
fprintf('HAEMORRHAGE V3 TRAINING COMPLETE\n');
fprintf('========================================\n');

fprintf('Saved: models/haemorrhagePatchNetV3.mat\n');