%% SeeBeyond - Haemorrhage Patch U-Net
% Corrected lesion-focused model

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

fprintf('\n===== HAEMORRHAGE PATCH U-NET =====\n');

%% Paths

patchRoot = fullfile(pwd, ...
    'datasets','IDRiD','patches','haemorrhages');

patchImageDir = fullfile(patchRoot,'images');
patchMaskDir  = fullfile(patchRoot,'masks');

%% Datastores

imdsHEPatch = imageDatastore(patchImageDir);

classNamesHEPatch = ["background","haemorrhage"];

pxdsHEPatch = pixelLabelDatastore( ...
    patchMaskDir, ...
    classNamesHEPatch, ...
    [0 1], ...
    FileExtensions=".png");

fprintf('Patch images : %d\n',numel(imdsHEPatch.Files));
fprintf('Patch masks  : %d\n',numel(pxdsHEPatch.Files));

%% Class statistics

countsHEPatch = countEachLabel(pxdsHEPatch);

disp(countsHEPatch)

freqHEPatch = ...
    countsHEPatch.PixelCount ./ sum(countsHEPatch.PixelCount);

fprintf('\nBackground frequency : %.6f\n',freqHEPatch(1));
fprintf('Haemorrhage frequency: %.6f\n',freqHEPatch(2));

%% Mild class weighting only
% Avoid excessive false positives.

suggestedWeightHE = ...
    sqrt(freqHEPatch(1)/max(freqHEPatch(2),eps));

haemorrhageWeight = min(1.5,suggestedWeightHE);

weightsHEPatch = [1; haemorrhageWeight];

fprintf('\nBackground weight : %.3f\n',weightsHEPatch(1));
fprintf('Haemorrhage weight: %.3f\n',weightsHEPatch(2));

classWeightsHEPatch = dlarray(weightsHEPatch,"C");

%% Combine

dsHEPatch = combine(imdsHEPatch,pxdsHEPatch);

%% Augmentation

dsHEPatchAug = transform(dsHEPatch,@augmentLesionData);

%% Verify sample

sampleHE = preview(dsHEPatchAug);

fprintf('\nImage size:\n');
disp(size(sampleHE{1}));

fprintf('Mask size:\n');
disp(size(sampleHE{2}));

fprintf('Undefined pixels: %d\n', ...
    sum(isundefined(sampleHE{2}(:))));

%% Lightweight U-Net

netHEPatch = unet( ...
    [256 256 3], ...
    2, ...
    EncoderDepth=3, ...
    NumFirstEncoderFilters=16);

%% Loss

lossHEPatch = @(Y,T) crossentropy( ...
    Y,T, ...
    classWeightsHEPatch, ...
    NormalizationFactor="all-elements");

%% Training options

optionsHEPatch = trainingOptions("adam", ...
    InitialLearnRate=1e-4, ...
    MaxEpochs=20, ...
    MiniBatchSize=4, ...
    Shuffle="every-epoch", ...
    ExecutionEnvironment="gpu", ...
    Plots="training-progress", ...
    Verbose=true);

%% Train

haemorrhagePatchNet = trainnet( ...
    dsHEPatchAug, ...
    netHEPatch, ...
    lossHEPatch, ...
    optionsHEPatch);

%% Save immediately

save('models/haemorrhagePatchNet.mat', ...
    'haemorrhagePatchNet', ...
    'weightsHEPatch', ...
    'freqHEPatch');

fprintf('\n========================================\n');
fprintf('HAEMORRHAGE PATCH TRAINING COMPLETE\n');
fprintf('Saved: models/haemorrhagePatchNet.mat\n');
fprintf('========================================\n');