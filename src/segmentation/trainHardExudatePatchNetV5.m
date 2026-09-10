%% SeeBeyond - Hard Exudate Patch Network V5
%
% 256x256 patch U-Net
% Weighted Cross-Entropy + Dice Loss
%
% Dataset:
%   hard_exudatesV5
%
% Model:
%   models/hardExudatePatchNetV5.mat

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

rng(42);

fprintf('\n=============================================\n');
fprintf(' SeeBeyond - Hard Exudate Patch Network V5\n');
fprintf('=============================================\n');

%% Paths

patchRoot = fullfile(pwd, ...
    'datasets','IDRiD','patches','hard_exudatesV5');

imageDir = fullfile(patchRoot,'images');
maskDir  = fullfile(patchRoot,'masks');

%% Datastores

imds = imageDatastore(imageDir);

classNames = ["background","hard_exudate"];
labelIDs = [0 1];

pxds = pixelLabelDatastore( ...
    maskDir, ...
    classNames, ...
    labelIDs, ...
    FileExtensions=".png");

fprintf('Training images/masks: %d\n',numel(imds.Files));

%% Verify pairing

if numel(imds.Files) ~= numel(pxds.Files)
    error('Image/mask count mismatch.');
end

%% Combined datastore
%
% MATLAB R2026a-compatible approach

ds = combine(imds,pxds);
%% Check sample

sample = preview(ds);

fprintf('\nSample datastore preview:\n');
disp(sample);

fprintf('Preview successful.\n');

%% Class distribution

tbl = countEachLabel(pxds);

disp(tbl);

totalPixels = sum(tbl.PixelCount);

frequency = tbl.PixelCount ./ totalPixels;

%% Moderate class weighting

weights = 1 ./ sqrt(frequency);

% Normalize background to 1
weights = weights ./ weights(1);

% Avoid excessive rare-class weighting
weights(2) = min(max(weights(2),1.5),4.0);

fprintf('\nClass weights:\n');
fprintf('Background   : %.4f\n',weights(1));
fprintf('Hard Exudate : %.4f\n',weights(2));

classWeights = dlarray(weights,"C");

%% U-Net

fprintf('\nCreating U-Net...\n');

net = unet( ...
    [256 256 3], ...
    2, ...
    EncoderDepth=3, ...
    NumFirstEncoderFilters=16);

summary(net);

%% Loss function

lossFcn = @(Y,T) hardExudateV5Loss( ...
    Y,T,classWeights);

%% Training options

options = trainingOptions("adam", ...
    InitialLearnRate=5e-5, ...
    MaxEpochs=40, ...
    MiniBatchSize=4, ...
    Shuffle="every-epoch", ...
    ExecutionEnvironment="gpu", ...
    GradientThreshold=1, ...
    Verbose=true, ...
    Plots="training-progress");

%% Train

fprintf('\n=============================================\n');
fprintf(' STARTING HARD EXUDATE V5 TRAINING\n');
fprintf('=============================================\n');

hardExudatePatchNetV5 = trainnet( ...
    ds, ...
    net, ...
    lossFcn, ...
    options);

%% Save

modelPath = fullfile( ...
    pwd,'models','hardExudatePatchNetV5.mat');

save(modelPath, ...
    'hardExudatePatchNetV5', ...
    'weights');

fprintf('\n=============================================\n');
fprintf(' HARD EXUDATE V5 TRAINING COMPLETE\n');
fprintf('=============================================\n');

fprintf('Model saved:\n%s\n',modelPath);

%% ============================================================
% Local loss function
% ============================================================

function loss = hardExudateV5Loss(Y,T,classWeights)

    % ---------------------------------------------------------
    % Weighted cross entropy
    % ---------------------------------------------------------

    ceLoss = crossentropy( ...
        Y, ...
        T, ...
        classWeights, ...
        NormalizationFactor="all-elements");

    % ---------------------------------------------------------
    % Extract foreground probabilities and labels
    % ---------------------------------------------------------

    P = Y(:,:,2,:);
    G = T(:,:,2,:);

    % ---------------------------------------------------------
    % Dice calculation
    % ---------------------------------------------------------

    epsilon = 1e-6;

    intersection = sum( ...
        sum(P .* G,1),2);

    predSum = sum( ...
        sum(P,1),2);

    gtSum = sum( ...
        sum(G,1),2);

    diceScore = ...
        (2 .* intersection + epsilon) ./ ...
        (predSum + gtSum + epsilon);

    % ---------------------------------------------------------
    % Only apply Dice loss to patches containing lesion
    % ---------------------------------------------------------

    hasLesion = gtSum > 0;

    diceLoss = sum( ...
        (1-diceScore) .* hasLesion) ./ ...
        (sum(hasLesion) + epsilon);

    % ---------------------------------------------------------
    % Combined loss
    % ---------------------------------------------------------

    loss = ...
        0.60 .* ceLoss + ...
        0.40 .* diceLoss;

end