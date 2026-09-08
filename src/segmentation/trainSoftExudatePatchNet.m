%% SeeBeyond - Soft Exudate Patch U-Net

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

fprintf('\n===== SOFT EXUDATE PATCH U-NET =====\n');

%% Load split
load('models/softExudateSplit.mat', ...
    'trainSEIdx','valSEIdx');

%% Paths
patchRoot = fullfile(pwd, ...
    'datasets','IDRiD','patches','soft_exudates');

patchImageDir = fullfile(patchRoot,'images');
patchMaskDir  = fullfile(patchRoot,'masks');

%% Datastores
imdsSEPatch = imageDatastore(patchImageDir);

classNamesSE = ["background","soft_exudate"];

pxdsSEPatch = pixelLabelDatastore( ...
    patchMaskDir, ...
    classNamesSE, ...
    [0 1], ...
    FileExtensions=".png");

fprintf('Patch images : %d\n',numel(imdsSEPatch.Files));
fprintf('Patch masks  : %d\n',numel(pxdsSEPatch.Files));

%% Class statistics
countsSE = countEachLabel(pxdsSEPatch);
disp(countsSE)

freqSE = countsSE.PixelCount ./ sum(countsSE.PixelCount);

fprintf('\nBackground frequency   : %.6f\n',freqSE(1));
fprintf('Soft Exudate frequency : %.6f\n',freqSE(2));

%% Mild class weighting
suggestedWeightSE = ...
    sqrt(freqSE(1)/max(freqSE(2),eps));

seWeight = min(1.5,suggestedWeightSE);

weightsSE = [1; seWeight];

fprintf('\nBackground weight   : %.3f\n',weightsSE(1));
fprintf('Soft Exudate weight : %.3f\n',weightsSE(2));

classWeightsSE = dlarray(weightsSE,"C");

%% Combine
dsSE = combine(imdsSEPatch,pxdsSEPatch);

%% Safe augmentation
dsSEAug = transform(dsSE,@augmentLesionData);

%% Verify sample
sampleSE = preview(dsSEAug);

fprintf('\nImage size:\n');
disp(size(sampleSE{1}));

fprintf('Mask size:\n');
disp(size(sampleSE{2}));

fprintf('Undefined pixels: %d\n', ...
    sum(isundefined(sampleSE{2}(:))));

%% U-Net
netSE = unet( ...
    [256 256 3], ...
    2, ...
    EncoderDepth=3, ...
    NumFirstEncoderFilters=16);

%% Weighted loss
lossSE = @(Y,T) crossentropy( ...
    Y,T, ...
    classWeightsSE, ...
    NormalizationFactor="all-elements");

%% Training
optionsSE = trainingOptions("adam", ...
    InitialLearnRate=1e-4, ...
    MaxEpochs=20, ...
    MiniBatchSize=4, ...
    Shuffle="every-epoch", ...
    ExecutionEnvironment="gpu", ...
    Plots="training-progress", ...
    Verbose=true);

softExudatePatchNet = trainnet( ...
    dsSEAug, ...
    netSE, ...
    lossSE, ...
    optionsSE);

%% Save
save('models/softExudatePatchNet.mat', ...
    'softExudatePatchNet', ...
    'weightsSE', ...
    'freqSE', ...
    'trainSEIdx', ...
    'valSEIdx');

fprintf('\n========================================\n');
fprintf('SOFT EXUDATE PATCH TRAINING COMPLETE\n');
fprintf('Saved: models/softExudatePatchNet.mat\n');
fprintf('========================================\n');