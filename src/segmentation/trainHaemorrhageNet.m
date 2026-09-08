%% SeeBeyond - Haemorrhage Segmentation

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

fprintf('\n===== HAEMORRHAGE SEGMENTATION =====\n');

%% Use same train/validation split
load('models/hardExudateNetV2.mat','trainEXIdx','valEXIdx');

trainHEIdx = trainEXIdx;
valHEIdx   = valEXIdx;

%% Dataset paths

idridRoot = fullfile(pwd, ...
    'datasets','IDRiD','Segmentation','A. Segmentation');

imageRoot = fullfile(idridRoot, ...
    '1. Original Images','a. Training Set');

heMaskRoot = fullfile(pwd, ...
    'datasets','IDRiD','processed_masks','haemorrhages');

%% Datastores

imdsHE = imageDatastore(imageRoot);

classNamesHE = ["background","haemorrhage"];

pxdsHE = pixelLabelDatastore( ...
    heMaskRoot, ...
    classNamesHE, ...
    [0 1], ...
    FileExtensions=".png");

fprintf('Images : %d\n',numel(imdsHE.Files));
fprintf('Masks  : %d\n',numel(pxdsHE.Files));

%% Train / validation

imdsHETrain = subset(imdsHE,trainHEIdx);
pxdsHETrain = subset(pxdsHE,trainHEIdx);

imdsHEVal = subset(imdsHE,valHEIdx);
pxdsHEVal = subset(pxdsHE,valHEIdx);

fprintf('Training   : %d\n',numel(imdsHETrain.Files));
fprintf('Validation : %d\n',numel(imdsHEVal.Files));

%% Class balance

tblHE = countEachLabel(pxdsHETrain);

disp(tblHE)

freqHE = tblHE.PixelCount ./ sum(tblHE.PixelCount);

suggestedWeight = sqrt(freqHE(1)/max(freqHE(2),eps));

% Mild weighting to avoid excessive false positives
lesionWeightHE = min(2.0,suggestedWeight);

weightsHE = [1; lesionWeightHE];

fprintf('\nBackground frequency : %.6f\n',freqHE(1));
fprintf('Haemorrhage frequency: %.6f\n',freqHE(2));

fprintf('Background weight    : %.3f\n',weightsHE(1));
fprintf('Haemorrhage weight   : %.3f\n',weightsHE(2));

classWeightsHE = dlarray(weightsHE,"C");

%% Combine

dsHETrain = combine(imdsHETrain,pxdsHETrain);
dsHEVal   = combine(imdsHEVal,pxdsHEVal);

%% Preprocess to 512x512

dsHETrain512 = transform(dsHETrain,@preprocessHaemorrhageData);
dsHEVal512   = transform(dsHEVal,@preprocessHaemorrhageData);

%% Augmentation

dsHETrainAug = transform(dsHETrain512,@augmentLesionData);

%% Verify sample

sampleHE = preview(dsHETrainAug);

fprintf('\nImage size:\n');
disp(size(sampleHE{1}));

fprintf('Mask size:\n');
disp(size(sampleHE{2}));

fprintf('Undefined pixels: %d\n', ...
    sum(isundefined(sampleHE{2}(:))));

%% Lightweight U-Net

netHE = unet( ...
    [512 512 3], ...
    2, ...
    EncoderDepth=3, ...
    NumFirstEncoderFilters=16);

%% Loss

lossHE = @(Y,T) crossentropy( ...
    Y,T, ...
    classWeightsHE, ...
    NormalizationFactor="all-elements");

%% Training options

optionsHE = trainingOptions("adam", ...
    InitialLearnRate=1e-4, ...
    MaxEpochs=40, ...
    MiniBatchSize=1, ...
    Shuffle="every-epoch", ...
    ExecutionEnvironment="gpu", ...
    Plots="training-progress", ...
    Verbose=true);

%% Train

haemorrhageNet = trainnet( ...
    dsHETrainAug, ...
    netHE, ...
    lossHE, ...
    optionsHE);

%% Save

save('models/haemorrhageNet.mat', ...
    'haemorrhageNet', ...
    'trainHEIdx', ...
    'valHEIdx', ...
    'weightsHE');

fprintf('\n====================================\n');
fprintf('HAEMORRHAGE TRAINING COMPLETE\n');
fprintf('Saved: models/haemorrhageNet.mat\n');
fprintf('====================================\n');