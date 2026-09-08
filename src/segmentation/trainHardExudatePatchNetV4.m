%% SeeBeyond - Hard Exudate Patch U-Net V4
% Hard-negative training + mild class weighting

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

fprintf('\n===== HARD EXUDATE PATCH U-NET V4 =====\n');

%% ------------------------------------------------
% Paths
%% ------------------------------------------------

v4Root = fullfile(pwd, ...
    'datasets','IDRiD','patches','hard_exudates_v4');

v4ImageDir = fullfile(v4Root,'images');
v4MaskDir  = fullfile(v4Root,'masks');

%% ------------------------------------------------
% Datastores
%% ------------------------------------------------

imdsV4 = imageDatastore(v4ImageDir);

classNamesV4 = ["background","hard_exudate"];

pxdsV4 = pixelLabelDatastore( ...
    v4MaskDir, ...
    classNamesV4, ...
    [0 1], ...
    FileExtensions=".png");

fprintf('V4 images : %d\n',numel(imdsV4.Files));
fprintf('V4 masks  : %d\n',numel(pxdsV4.Files));

%% ------------------------------------------------
% Class statistics
%% ------------------------------------------------

countsV4 = countEachLabel(pxdsV4);

disp(countsV4)

totalPixelsV4 = sum(countsV4.PixelCount);

freqV4 = countsV4.PixelCount ./ totalPixelsV4;

fprintf('\nBackground frequency : %.4f\n',freqV4(1));
fprintf('Exudate frequency    : %.4f\n',freqV4(2));

%% ------------------------------------------------
% IMPORTANT:
% Mild weighting only.
%
% V3 used ~3.42x lesion weight and produced too many
% false positives.
%
% V4 caps lesion weight at 1.5.
%% ------------------------------------------------

suggestedWeight = sqrt(freqV4(1) / max(freqV4(2),eps));

lesionWeight = min(1.5,suggestedWeight);

weightsV4 = [1; lesionWeight];

fprintf('\nBackground weight : %.3f\n',weightsV4(1));
fprintf('Exudate weight    : %.3f\n',weightsV4(2));

classWeightsV4 = dlarray(weightsV4,"C");

%% ------------------------------------------------
% Combine data
%% ------------------------------------------------

dsV4 = combine(imdsV4,pxdsV4);

%% Safe augmentation
dsV4Aug = transform(dsV4,@augmentLesionData);

%% ------------------------------------------------
% Verify data
%% ------------------------------------------------

sampleV4 = preview(dsV4Aug);

fprintf('\nImage size:\n');
disp(size(sampleV4{1}));

fprintf('Mask size:\n');
disp(size(sampleV4{2}));

fprintf('Undefined pixels: %d\n', ...
    sum(isundefined(sampleV4{2}(:))));

%% ------------------------------------------------
% U-Net
%% ------------------------------------------------

netV4 = unet( ...
    [256 256 3], ...
    2, ...
    EncoderDepth=3, ...
    NumFirstEncoderFilters=16);

summary(netV4)

%% ------------------------------------------------
% Mild weighted cross entropy
%% ------------------------------------------------

lossV4 = @(Y,T) crossentropy( ...
    Y,T, ...
    classWeightsV4, ...
    NormalizationFactor="all-elements");

%% ------------------------------------------------
% Training
%% ------------------------------------------------

optionsV4 = trainingOptions("adam", ...
    InitialLearnRate=1e-4, ...
    MaxEpochs=20, ...
    MiniBatchSize=4, ...
    Shuffle="every-epoch", ...
    ExecutionEnvironment="gpu", ...
    Plots="training-progress", ...
    Verbose=true);

hardExudatePatchNetV4 = trainnet( ...
    dsV4Aug, ...
    netV4, ...
    lossV4, ...
    optionsV4);

%% ------------------------------------------------
% Save immediately
%% ------------------------------------------------

save('models/hardExudatePatchNetV4.mat', ...
    'hardExudatePatchNetV4', ...
    'weightsV4', ...
    'freqV4');

fprintf('\n========================================\n');
fprintf('HARD EXUDATE V4 TRAINING COMPLETE\n');
fprintf('Saved: models/hardExudatePatchNetV4.mat\n');
fprintf('========================================\n');