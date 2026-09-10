%% SeeBeyond - Microaneurysm Patch Model Validation
% Validates the EXISTING trained model on the 11 held-out IDRiD images.
% No training is performed.

clear;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

fprintf('\n==============================================\n');
fprintf('MICROANEURYSM PATCH MODEL VALIDATION\n');
fprintf('==============================================\n');

%% ------------------------------------------------
% Load trained model and validation split
%% ------------------------------------------------

modelFile = 'models/microaneurysmPatchNet.mat';

if ~isfile(modelFile)
    error('Microaneurysm model not found: %s', modelFile);
end

S = load(modelFile, ...
    'microaneurysmPatchNet', ...
    'valMAIdx');

net = S.microaneurysmPatchNet;
valMAIdx = S.valMAIdx;

fprintf('Validation images: %d\n', numel(valMAIdx));

%% ------------------------------------------------
% Paths
%% ------------------------------------------------

idridRoot = fullfile(pwd, ...
    'datasets','IDRiD','Segmentation','A. Segmentation');

imageRoot = fullfile(idridRoot, ...
    '1. Original Images','a. Training Set');

maMaskRoot = fullfile(pwd, ...
    'datasets','IDRiD','processed_masks','microaneurysms');

imds = imageDatastore(imageRoot);

%% ------------------------------------------------
% Validation parameters
%% ------------------------------------------------

patchSize = 256;

% Number of validation patches per image.
% These are generated ONLY from the held-out images.
positivePatchesPerImage = 20;
negativePatchesPerImage = 20;

rng(42);

%% ------------------------------------------------
% Metric accumulators
%% ------------------------------------------------

TP = 0;
TN = 0;
FP = 0;
FN = 0;

patchCount = 0;

imageDice = [];
imageIoU = [];
imageSensitivity = [];
imageSpecificity = [];

%% ------------------------------------------------
% Process validation images
%% ------------------------------------------------

for n = 1:numel(valMAIdx)

    idx = valMAIdx(n);

    I = readimage(imds,idx);

    [~,imageID,~] = fileparts(imds.Files{idx});

    maPath = fullfile(maMaskRoot,[imageID '.png']);

    if ~isfile(maPath)
        error('Microaneurysm mask not found: %s',maPath);
    end

    MA = imread(maPath) > 0;

    [h,w,~] = size(I);

    %% Retina field

    G = im2double(rgb2gray(I));

    retinaMask = G > 0.03;
    retinaMask = bwareafilt(retinaMask,1);
    retinaMask = imfill(retinaMask,'holes');

    %% Per-image metric accumulators

    TPimg = 0;
    TNimg = 0;
    FPimg = 0;
    FNimg = 0;

    %% ============================================
    % POSITIVE VALIDATION PATCHES
    %% ============================================

    [rr,cc] = find(MA);

    if ~isempty(rr)

        for p = 1:positivePatchesPerImage

            k = randi(numel(rr));

            centerR = rr(k) + randi([-48 48]);
            centerC = cc(k) + randi([-48 48]);

            r1 = round(centerR - patchSize/2);
            c1 = round(centerC - patchSize/2);

            r1 = max(1,min(r1,h-patchSize+1));
            c1 = max(1,min(c1,w-patchSize+1));

            rows = r1:r1+patchSize-1;
            cols = c1:c1+patchSize-1;

            Ipatch = I(rows,cols,:);
            Mpatch = MA(rows,cols);

            [predMask,scoreMap] = predictMAPatch(net,Ipatch);

            %#ok<NASGU>
            scoreMap = scoreMap;

            TPp = nnz(predMask & Mpatch);
            TNp = nnz(~predMask & ~Mpatch);
            FPp = nnz(predMask & ~Mpatch);
            FNp = nnz(~predMask & Mpatch);

            TP = TP + TPp;
            TN = TN + TNp;
            FP = FP + FPp;
            FN = FN + FNp;

            TPimg = TPimg + TPp;
            TNimg = TNimg + TNp;
            FPimg = FPimg + FPp;
            FNimg = FNimg + FNp;

            patchCount = patchCount + 1;
        end

    end

    %% ============================================
    % NEGATIVE VALIDATION PATCHES
    %% ============================================

    createdNeg = 0;
    attempts = 0;

    while createdNeg < negativePatchesPerImage && ...
            attempts < 1000

        attempts = attempts + 1;

        centerR = randi(h);
        centerC = randi(w);

        r1 = round(centerR - patchSize/2);
        c1 = round(centerC - patchSize/2);

        r1 = max(1,min(r1,h-patchSize+1));
        c1 = max(1,min(c1,w-patchSize+1));

        rows = r1:r1+patchSize-1;
        cols = c1:c1+patchSize-1;

        retinaPatch = retinaMask(rows,cols);
        Mpatch = MA(rows,cols);

        % Require mostly retina and no true MA pixels.
        if mean(retinaPatch(:)) > 0.70 && nnz(Mpatch) == 0

            Ipatch = I(rows,cols,:);

            [predMask,scoreMap] = predictMAPatch(net,Ipatch);

            %#ok<NASGU>
            scoreMap = scoreMap;

            TPp = nnz(predMask & Mpatch);
            TNp = nnz(~predMask & ~Mpatch);
            FPp = nnz(predMask & ~Mpatch);
            FNp = nnz(~predMask & Mpatch);

            TP = TP + TPp;
            TN = TN + TNp;
            FP = FP + FPp;
            FN = FN + FNp;

            TPimg = TPimg + TPp;
            TNimg = TNimg + TNp;
            FPimg = FPimg + FPp;
            FNimg = FNimg + FNp;

            createdNeg = createdNeg + 1;
            patchCount = patchCount + 1;
        end
    end

    %% Per-image metrics

    dice = (2*TPimg) / max(2*TPimg + FPimg + FNimg,eps);

    iou = TPimg / max(TPimg + FPimg + FNimg,eps);

    sensitivity = TPimg / max(TPimg + FNimg,eps);

    specificity = TNimg / max(TNimg + FPimg,eps);

    imageDice(end+1) = dice;
    imageIoU(end+1) = iou;
    imageSensitivity(end+1) = sensitivity;
    imageSpecificity(end+1) = specificity;

    fprintf(['Image %s: Dice=%.4f, IoU=%.4f, ' ...
        'Sensitivity=%.4f, Specificity=%.4f\n'], ...
        imageID,dice,iou,sensitivity,specificity);
end

%% ------------------------------------------------
% Overall metrics
%% ------------------------------------------------

meanDice = mean(imageDice);
meanIoU = mean(imageIoU);
meanSensitivity = mean(imageSensitivity);
meanSpecificity = mean(imageSpecificity);

fprintf('\n==============================================\n');
fprintf('MICROANEURYSM VALIDATION COMPLETE\n');
fprintf('==============================================\n');

fprintf('Validation images : %d\n',numel(valMAIdx));
fprintf('Validation patches: %d\n',patchCount);

fprintf('\nMean Dice        : %.4f\n',meanDice);
fprintf('Mean IoU         : %.4f\n',meanIoU);
fprintf('Mean Sensitivity : %.4f\n',meanSensitivity);
fprintf('Mean Specificity : %.4f\n',meanSpecificity);

fprintf('\nPixel-level totals:\n');
fprintf('TP = %d\n',TP);
fprintf('TN = %d\n',TN);
fprintf('FP = %d\n',FP);
fprintf('FN = %d\n',FN);

%% ------------------------------------------------
% Save metrics
%% ------------------------------------------------

results.microaneurysm = struct( ...
    'meanDice',meanDice, ...
    'meanIoU',meanIoU, ...
    'meanSensitivity',meanSensitivity, ...
    'meanSpecificity',meanSpecificity, ...
    'TP',TP, ...
    'TN',TN, ...
    'FP',FP, ...
    'FN',FN, ...
    'validationImages',numel(valMAIdx), ...
    'validationPatches',patchCount);

results.perImage.dice = imageDice;
results.perImage.iou = imageIoU;
results.perImage.sensitivity = imageSensitivity;
results.perImage.specificity = imageSpecificity;

if ~exist('results','dir')
    mkdir('results');
end

save('results/microaneurysmValidation.mat','results');

fprintf('\nSaved:\n');
fprintf('results/microaneurysmValidation.mat\n');
fprintf('==============================================\n');

%% =================================================
% Local prediction function
%% =================================================

function [mask,scoreMap] = predictMAPatch(net,Ipatch)

    % Convert uint8 RGB image to single precision.
    X = single(Ipatch);

% IMPORTANT:
% The Microaneurysm network uses an ImageInputLayer
% with zerocenter normalization.
% Therefore DO NOT divide the image by 255.
% The network input layer handles the normalization.

Y = predict(net,X);

    % Remove batch dimension.
    Y = extractdata(Y);

    Y = squeeze(Y);

    % Network output should be:
    % H x W x 2
    if ndims(Y) ~= 3 || size(Y,3) ~= 2
        error('Unexpected Microaneurysm network output size.');
    end

    % Class 2 = microaneurysm.
    scoreMap = Y(:,:,2);

    % Threshold.
    mask = scoreMap >= 0.50;
end