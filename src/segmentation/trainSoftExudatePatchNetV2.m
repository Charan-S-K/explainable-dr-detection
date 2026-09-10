%% SeeBeyond - Soft Exudate Patch U-Net V2
% Improved Soft Exudate segmentation

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

rng(42);

fprintf('\n========================================\n');
fprintf('SOFT EXUDATE PATCH U-NET V2\n');
fprintf('========================================\n');

%% Load split

load('models/softExudateV2Split.mat', ...
    'trainSEIdx','valSEIdx', ...
    'positiveImageCount', ...
    'positiveCount', ...
    'hardExNegCount', ...
    'opticDiscNegCount', ...
    'normalNegCount');

%% Paths

patchRoot = fullfile(pwd, ...
    'datasets','IDRiD','patches','soft_exudates_v2');

imageDir = fullfile(patchRoot,'images');
maskDir  = fullfile(patchRoot,'masks');

%% Datastores

imdsSEV2 = imageDatastore(imageDir);

classNamesSEV2 = [
    "background"
    "soft_exudate"
];

pxdsSEV2 = pixelLabelDatastore( ...
    maskDir, ...
    classNamesSEV2, ...
    [0 1], ...
    FileExtensions=".png");

fprintf('Images : %d\n',numel(imdsSEV2.Files));
fprintf('Masks  : %d\n',numel(pxdsSEV2.Files));

%% Pixel statistics

countsSEV2 = countEachLabel(pxdsSEV2);

fprintf('\n===== PIXEL DISTRIBUTION =====\n');
disp(countsSEV2)

freqSEV2 = ...
    countsSEV2.PixelCount ./ ...
    sum(countsSEV2.PixelCount);

fprintf('Background frequency   : %.6f\n',freqSEV2(1));
fprintf('Soft Exudate frequency : %.6f\n',freqSEV2(2));

%% Combine + augmentation

dsSEV2 = combine( ...
    imdsSEV2, ...
    pxdsSEV2);

dsSEV2 = transform( ...
    dsSEV2, ...
    @augmentLesionData);

%% Verify sample

sample = preview(dsSEV2);

fprintf('\nImage size:\n');
disp(size(sample{1}));

fprintf('Mask size:\n');
disp(size(sample{2}));

fprintf('Undefined pixels : %d\n', ...
    sum(isundefined(sample{2}(:))));

%% U-Net

netSEV2 = unet( ...
    [256 256 3], ...
    2, ...
    EncoderDepth=3, ...
    NumFirstEncoderFilters=16);

%% Training options

optionsSEV2 = trainingOptions("adam", ...
    InitialLearnRate=5e-5, ...
    MaxEpochs=30, ...
    MiniBatchSize=4, ...
    Shuffle="every-epoch", ...
    ExecutionEnvironment="gpu", ...
    Plots="training-progress", ...
    Verbose=true);

%% Train

fprintf('\nStarting Soft Exudate V2 training...\n');

softExudatePatchNetV2 = trainnet( ...
    dsSEV2, ...
    netSEV2, ...
    @softExudateV2Loss, ...
    optionsSEV2);

%% Save separately

save('models/softExudatePatchNetV2.mat', ...
    'softExudatePatchNetV2', ...
    'freqSEV2', ...
    'trainSEIdx', ...
    'valSEIdx');

fprintf('\n========================================\n');
fprintf('SOFT EXUDATE V2 TRAINING COMPLETE\n');
fprintf('========================================\n');

fprintf('Saved: models/softExudatePatchNetV2.mat\n');
