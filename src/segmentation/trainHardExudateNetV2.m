%% SeeBeyond - Hard Exudate Model V2

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');

addpath(genpath(fullfile(pwd,'src')));

fprintf('\n===== SeeBeyond Hard Exudate V2 =====\n');

%% Dataset paths

idridRoot = fullfile(pwd, ...
    'datasets','IDRiD','Segmentation','A. Segmentation');

trainImageRoot = fullfile(idridRoot, ...
    '1. Original Images','a. Training Set');

exProcessed = fullfile(pwd, ...
    'datasets','IDRiD','processed_masks','hard_exudates');

%% Datastores

idridImages = imageDatastore(trainImageRoot);

classNamesEX = ["background","hard_exudate"];

pxdsEX = pixelLabelDatastore( ...
    exProcessed, ...
    classNamesEX, ...
    [0 1], ...
    FileExtensions=".png");

fprintf('Images : %d\n',numel(idridImages.Files));
fprintf('Masks  : %d\n',numel(pxdsEX.Files));

%% Same reproducible split

rng(42);

idx = randperm(numel(idridImages.Files));

trainEXIdx = idx(1:43);
valEXIdx   = idx(44:54);

imdsEXTrain = subset(idridImages,trainEXIdx);
pxdsEXTrain = subset(pxdsEX,trainEXIdx);

imdsEXVal = subset(idridImages,valEXIdx);
pxdsEXVal = subset(pxdsEX,valEXIdx);

fprintf('Training   : %d\n',numel(imdsEXTrain.Files));
fprintf('Validation : %d\n',numel(imdsEXVal.Files));

%% Determine class imbalance using TRAINING data only

tblEX = countEachLabel(pxdsEXTrain);

disp(tblEX)

totalPixels = sum(tblEX.PixelCount);

frequency = tblEX.PixelCount ./ totalPixels;

% Moderate inverse-frequency weighting.
% sqrt prevents the rare-class weight becoming excessively large.
weights = 1 ./ sqrt(frequency);

% Make background weight = 1
weights = weights ./ weights(1);

fprintf('\nBackground weight    : %.3f\n',weights(1));
fprintf('Hard Exudate weight  : %.3f\n',weights(2));

classWeightsEX = dlarray(weights,"C");

%% Combine image and mask data

dsEXTrain = combine(imdsEXTrain,pxdsEXTrain);
dsEXVal   = combine(imdsEXVal,pxdsEXVal);

%% 512x512 preprocessing

dsEXTrain512 = transform( ...
    dsEXTrain,@preprocessHardExudateData);

dsEXVal512 = transform( ...
    dsEXVal,@preprocessHardExudateData);

%% Safe augmentation

dsEXTrainAug = transform( ...
    dsEXTrain512,@augmentLesionData);

%% Verify training sample

sample = preview(dsEXTrainAug);

fprintf('\nInput size:\n');
disp(size(sample{1}));

fprintf('Mask size:\n');
disp(size(sample{2}));

fprintf('Undefined mask pixels: %d\n', ...
    sum(isundefined(sample{2}(:))));

%% Smaller U-Net
% Better suited to RTX 3050 6 GB at 512x512.

netEXv2 = unet( ...
    [512 512 3], ...
    2, ...
    EncoderDepth=3, ...
    NumFirstEncoderFilters=16);

summary(netEXv2)

%% Weighted Cross Entropy

lossFcnEX = @(Y,T) crossentropy( ...
    Y,T, ...
    classWeightsEX, ...
    NormalizationFactor="all-elements");

%% Training options

optionsEXv2 = trainingOptions("adam", ...
    InitialLearnRate=1e-4, ...
    MaxEpochs=50, ...
    MiniBatchSize=1, ...
    Shuffle="every-epoch", ...
    ExecutionEnvironment="gpu", ...
    Plots="training-progress", ...
    Verbose=true);

%% TRAIN

hardExudateNetV2 = trainnet( ...
    dsEXTrainAug, ...
    netEXv2, ...
    lossFcnEX, ...
    optionsEXv2);

%% SAVE IMMEDIATELY

save('models/hardExudateNetV2.mat', ...
    'hardExudateNetV2', ...
    'trainEXIdx', ...
    'valEXIdx', ...
    'weights');

disp(' ');
disp('======================================');
disp('Hard Exudate V2 training COMPLETE');
disp('Saved: models/hardExudateNetV2.mat');
disp('======================================');