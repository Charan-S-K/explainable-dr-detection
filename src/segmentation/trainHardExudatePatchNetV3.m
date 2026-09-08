%% SeeBeyond - Hard Exudate Patch U-Net V3

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

fprintf('\n===== Hard Exudate Patch Model V3 =====\n');

%% Patch folders

patchRoot = fullfile(pwd, ...
    'datasets','IDRiD','patches','hard_exudates');

patchImageDir = fullfile(patchRoot,'images');
patchMaskDir  = fullfile(patchRoot,'masks');

%% Datastores

patchImds = imageDatastore(patchImageDir);

classNamesV3 = ["background","hard_exudate"];

patchPxds = pixelLabelDatastore( ...
    patchMaskDir, ...
    classNamesV3, ...
    [0 1], ...
    FileExtensions=".png");

fprintf('Patch images : %d\n',numel(patchImds.Files));
fprintf('Patch masks  : %d\n',numel(patchPxds.Files));

%% Check class balance

patchCounts = countEachLabel(patchPxds);
disp(patchCounts)

totalPixels = sum(patchCounts.PixelCount);
freq = patchCounts.PixelCount ./ totalPixels;

weightsV3 = 1 ./ sqrt(freq);
weightsV3 = weightsV3 ./ weightsV3(1);

fprintf('Background frequency : %.4f\n',freq(1));
fprintf('Exudate frequency    : %.4f\n',freq(2));

fprintf('Background weight    : %.3f\n',weightsV3(1));
fprintf('Exudate weight       : %.3f\n',weightsV3(2));

classWeightsV3 = dlarray(weightsV3,"C");

%% Combine data

dsPatch = combine(patchImds,patchPxds);

%% Safe augmentation

dsPatchAug = transform(dsPatch,@augmentLesionData);

%% Verify one sample

sampleV3 = preview(dsPatchAug);

fprintf('Image size:\n');
disp(size(sampleV3{1}));

fprintf('Mask size:\n');
disp(size(sampleV3{2}));

fprintf('Undefined pixels: %d\n', ...
    sum(isundefined(sampleV3{2}(:))));

%% Lightweight U-Net

netEXv3 = unet( ...
    [256 256 3], ...
    2, ...
    EncoderDepth=3, ...
    NumFirstEncoderFilters=16);

%% Weighted loss

lossV3 = @(Y,T) crossentropy( ...
    Y,T, ...
    classWeightsV3, ...
    NormalizationFactor="all-elements");

%% Training options

optionsV3 = trainingOptions("adam", ...
    InitialLearnRate=1e-4, ...
    MaxEpochs=20, ...
    MiniBatchSize=4, ...
    Shuffle="every-epoch", ...
    ExecutionEnvironment="gpu", ...
    Plots="training-progress", ...
    Verbose=true);

%% Train

hardExudatePatchNetV3 = trainnet( ...
    dsPatchAug, ...
    netEXv3, ...
    lossV3, ...
    optionsV3);

%% Save immediately

save('models/hardExudatePatchNetV3.mat', ...
    'hardExudatePatchNetV3', ...
    'weightsV3');

fprintf('\n====================================\n');
fprintf('V3 PATCH MODEL TRAINING COMPLETE\n');
fprintf('Saved: models/hardExudatePatchNetV3.mat\n');
fprintf('====================================\n');