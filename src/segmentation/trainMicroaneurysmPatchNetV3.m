%% SeeBeyond - Microaneurysm V3 Patch U-Net
%
% V3 training objective:
%   60% weighted cross-entropy
%   40% soft Dice loss
%
% Main purpose:
%   Recover the sensitivity lost by V2 while maintaining
%   reasonable false-positive suppression.
%
% V1 and V2 models are preserved.

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

fprintf('\n');
fprintf('====================================================\n');
fprintf('MICROANEURYSM V3 PATCH U-NET TRAINING\n');
fprintf('====================================================\n');

%% ---------------------------------------------------
% Load V3 split
% ----------------------------------------------------

splitFile = 'models/microaneurysmV3Split.mat';

if ~isfile(splitFile)

    error( ...
        ['V3 split not found.\n' ...
         'Run createMicroaneurysmPatchesV3.m first.']);

end

load( ...
    splitFile, ...
    'trainMAIdx', ...
    'valMAIdx');

fprintf('Training images   : %d\n',numel(trainMAIdx));
fprintf('Validation images : %d\n',numel(valMAIdx));

%% ---------------------------------------------------
% Patch paths
% ----------------------------------------------------

patchRoot = fullfile( ...
    pwd, ...
    'datasets','IDRiD','patches','microaneurysmsV3');

patchImageDir = fullfile( ...
    patchRoot,'images');

patchMaskDir = fullfile( ...
    patchRoot,'masks');

if ~isfolder(patchImageDir)

    error( ...
        'V3 patch image directory not found: %s', ...
        patchImageDir);

end

if ~isfolder(patchMaskDir)

    error( ...
        'V3 patch mask directory not found: %s', ...
        patchMaskDir);

end

%% ---------------------------------------------------
% Datastores
% ----------------------------------------------------

imdsMAPatch = imageDatastore( ...
    patchImageDir);

classNamesMA = [ ...
    "background", ...
    "microaneurysm"];

pxdsMAPatch = pixelLabelDatastore( ...
    patchMaskDir, ...
    classNamesMA, ...
    [0 1], ...
    FileExtensions=".png");

fprintf('\nPatch images : %d\n', ...
    numel(imdsMAPatch.Files));

fprintf('Patch masks  : %d\n', ...
    numel(pxdsMAPatch.Files));

%% ---------------------------------------------------
% Verify datastore
% ----------------------------------------------------

if numel(imdsMAPatch.Files) ~= ...
        numel(pxdsMAPatch.Files)

    error( ...
        'Number of patch images and masks does not match.');

end

%% ---------------------------------------------------
% Class frequency
% ----------------------------------------------------

countsMA = countEachLabel( ...
    pxdsMAPatch);

fprintf('\nPixel class counts:\n');
disp(countsMA);

freqMA = countsMA.PixelCount ./ ...
    sum(countsMA.PixelCount);

fprintf('\nBackground frequency    : %.8f\n', ...
    freqMA(1));

fprintf('Microaneurysm frequency : %.8f\n', ...
    freqMA(2));

%% ---------------------------------------------------
% V3 class weights
% ----------------------------------------------------
%
% We deliberately avoid an extremely large MA weight.
% The goal is to improve sensitivity without making the
% model produce excessive false positives.

suggestedWeightMA = sqrt( ...
    freqMA(1) / max(freqMA(2),eps));

maWeight = min(3.0, ...
    max(1.5,suggestedWeightMA));

weightsMA = [ ...
    1; ...
    maWeight];

fprintf('\nV3 class weights:\n');

fprintf('Background weight    : %.3f\n', ...
    weightsMA(1));

fprintf('Microaneurysm weight : %.3f\n', ...
    weightsMA(2));

classWeightsMA = dlarray( ...
    weightsMA, ...
    "C");

%% ---------------------------------------------------
% Combine image + label datastores
% ----------------------------------------------------

dsMA = combine( ...
    imdsMAPatch, ...
    pxdsMAPatch);

%% ---------------------------------------------------
% Data augmentation
% ----------------------------------------------------
%
% Only flips are used.
% We avoid arbitrary rotations because the previous
% segmentation pipeline uses pixel-aligned masks and
% we want conservative augmentation.

dsMAAug = transform( ...
    dsMA, ...
    @augmentMicroaneurysmV3Data);

%% ---------------------------------------------------
% Check one sample
% ----------------------------------------------------

sampleMA = preview(dsMAAug);

fprintf('\nSample image size:\n');
disp(size(sampleMA{1}));

fprintf('Sample mask size:\n');
disp(size(sampleMA{2}));

undefinedCount = sum( ...
    isundefined(sampleMA{2}(:)));

fprintf('Undefined pixels: %d\n', ...
    undefinedCount);

if undefinedCount > 0

    error( ...
        'Undefined pixels found in V3 sample mask.');

end

%% ---------------------------------------------------
% Build U-Net
% ----------------------------------------------------

fprintf('\nBuilding V3 U-Net...\n');

netMA = unet( ...
    [256 256 3], ...
    2, ...
    EncoderDepth=3, ...
    NumFirstEncoderFilters=16);

fprintf('V3 U-Net created.\n');

%% ---------------------------------------------------
% V3 combined loss
% ----------------------------------------------------
%
% Loss = 0.60 Cross Entropy
%      + 0.40 Soft Dice
%
% Cross entropy maintains pixel-level discrimination.
% Dice directly encourages overlap with small lesions.

lossMA = @(Y,T) microaneurysmV3Loss( ...
    Y, ...
    T, ...
    classWeightsMA);

%% ---------------------------------------------------
% Training options
% ----------------------------------------------------

optionsMA = trainingOptions( ...
    "adam", ...
    InitialLearnRate=5e-5, ...
    MaxEpochs=35, ...
    MiniBatchSize=4, ...
    Shuffle="every-epoch", ...
    ExecutionEnvironment="gpu", ...
    Plots="training-progress", ...
    Verbose=true);

fprintf('\nTraining settings:\n');

fprintf('Optimizer          : Adam\n');
fprintf('Initial LR         : 5e-5\n');
fprintf('Maximum epochs     : 35\n');
fprintf('Mini-batch size    : 4\n');
fprintf('Execution          : GPU\n');
fprintf('Loss               : 0.60 CE + 0.40 Dice\n');

%% ---------------------------------------------------
% Train
% ----------------------------------------------------

fprintf('\n');
fprintf('====================================================\n');
fprintf('STARTING MICROANEURYSM V3 TRAINING\n');
fprintf('====================================================\n');

microaneurysmPatchNetV3 = trainnet( ...
    dsMAAug, ...
    netMA, ...
    lossMA, ...
    optionsMA);

%% ---------------------------------------------------
% Save model
% ----------------------------------------------------

save( ...
    'models/microaneurysmPatchNetV3.mat', ...
    'microaneurysmPatchNetV3', ...
    'weightsMA', ...
    'freqMA', ...
    'trainMAIdx', ...
    'valMAIdx');

fprintf('\n');
fprintf('====================================================\n');
fprintf('MICROANEURYSM V3 TRAINING COMPLETE\n');
fprintf('====================================================\n');

fprintf('Saved model:\n');
fprintf('models/microaneurysmPatchNetV3.mat\n');

fprintf('====================================================\n');


%% ===================================================
% LOCAL FUNCTIONS
% ====================================================

function dataOut = augmentMicroaneurysmV3Data(data)

    I = data{1};
    C = data{2};

    % Horizontal flip
    if rand > 0.5

        I = fliplr(I);
        C = fliplr(C);

    end

    % Vertical flip
    if rand > 0.5

        I = flipud(I);
        C = flipud(C);

    end

    dataOut = {I,C};

end


function loss = microaneurysmV3Loss( ...
    Y, ...
    T, ...
    classWeights)

    %% -----------------------------------------------
    % Weighted cross entropy
    % -----------------------------------------------

    ceLoss = crossentropy( ...
        Y, ...
        T, ...
        classWeights, ...
        NormalizationFactor="all-elements");

    %% -----------------------------------------------
    % Microaneurysm probability
    % -----------------------------------------------

    P = Y(:,:,2,:);

    G = T(:,:,2,:);

    %% -----------------------------------------------
    % Soft Dice
    % -----------------------------------------------

    intersection = sum( ...
        sum(P .* G,1), ...
        2);

    predSum = sum( ...
        sum(P,1), ...
        2);

    gtSum = sum( ...
        sum(G,1), ...
        2);

    epsilon = 1e-6;

    diceScore = ...
        (2 .* intersection + epsilon) ./ ...
        (predSum + gtSum + epsilon);

    %% -----------------------------------------------
    % Only lesion-containing patches contribute
    % strongly to Dice component.
    % -----------------------------------------------

    hasLesion = gtSum > 0;

    diceLoss = sum( ...
        (1 - diceScore) .* hasLesion) ./ ...
        (sum(hasLesion) + epsilon);

    %% -----------------------------------------------
    % Combined loss
    % -----------------------------------------------

    loss = ...
        0.60 .* ceLoss + ...
        0.40 .* diceLoss;

end