%% SeeBeyond - Microaneurysm V3 Patch Model Validation
%
% Validates V3 on the SAME held-out validation images and
% using the SAME sampling procedure as V1 and V2.
%
% This is a fair comparison:
%
% V1 = microaneurysmPatchNet.mat
% V2 = microaneurysmPatchNetV2.mat
% V3 = microaneurysmPatchNetV3.mat
%
% No training is performed.

clear;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

fprintf('\n');
fprintf('====================================================\n');
fprintf('MICROANEURYSM V3 PATCH MODEL VALIDATION\n');
fprintf('====================================================\n');

%% ---------------------------------------------------
% Load V3 model
% ----------------------------------------------------

modelFile = 'models/microaneurysmPatchNetV3.mat';

if ~isfile(modelFile)

    error( ...
        'Microaneurysm V3 model not found: %s', ...
        modelFile);

end

S = load( ...
    modelFile, ...
    'microaneurysmPatchNetV3', ...
    'valMAIdx');

net = S.microaneurysmPatchNetV3;

valMAIdx = S.valMAIdx;

fprintf('Validation images: %d\n', ...
    numel(valMAIdx));

%% ---------------------------------------------------
% Paths
% ----------------------------------------------------

idridRoot = fullfile( ...
    pwd, ...
    'datasets','IDRiD','Segmentation','A. Segmentation');

imageRoot = fullfile( ...
    idridRoot, ...
    '1. Original Images','a. Training Set');

maMaskRoot = fullfile( ...
    pwd, ...
    'datasets','IDRiD','processed_masks','microaneurysms');

imds = imageDatastore(imageRoot);

%% ---------------------------------------------------
% Validation settings
% ----------------------------------------------------

patchSize = 256;

positivePatchesPerImage = 20;

negativePatchesPerImage = 20;

threshold = 0.50;

% IMPORTANT:
% Same random seed as V1 and V2.
rng(42);

fprintf('\nValidation settings:\n');

fprintf('Patch size       : %d x %d\n', ...
    patchSize,patchSize);

fprintf('Positive patches : %d / image\n', ...
    positivePatchesPerImage);

fprintf('Negative patches : %d / image\n', ...
    negativePatchesPerImage);

fprintf('Threshold        : %.2f\n', ...
    threshold);

%% ---------------------------------------------------
% Global metrics
% ----------------------------------------------------

TP = 0;
TN = 0;
FP = 0;
FN = 0;

patchCount = 0;

%% Per-image metrics
% ----------------------------------------------------

imageDice = [];
imageIoU = [];
imageSensitivity = [];
imageSpecificity = [];

imageTP = [];
imageTN = [];
imageFP = [];
imageFN = [];

%% ---------------------------------------------------
% Validation loop
% ----------------------------------------------------

for n = 1:numel(valMAIdx)

    idx = valMAIdx(n);

    I = readimage( ...
        imds, ...
        idx);

    [~,imageID,~] = ...
        fileparts(imds.Files{idx});

    %% Load ground truth
    maPath = fullfile( ...
        maMaskRoot, ...
        [imageID '.png']);

    if ~isfile(maPath)

        error( ...
            'Microaneurysm mask not found: %s', ...
            maPath);

    end

    MA = imread(maPath) > 0;

    [h,w,~] = size(I);

    %% Retina mask
    G = im2double( ...
        rgb2gray(I));

    retinaMask = G > 0.03;

    if any(retinaMask(:))

        retinaMask = bwareafilt( ...
            retinaMask,1);

        retinaMask = imfill( ...
            retinaMask,'holes');

    end

    %% Image-level counters
    TPimg = 0;
    TNimg = 0;
    FPimg = 0;
    FNimg = 0;

    %% ------------------------------------------------
    % Positive patches
    % -------------------------------------------------

    [rr,cc] = find(MA);

    if ~isempty(rr)

        for p = 1:positivePatchesPerImage

            k = randi(numel(rr));

            centerR = rr(k) + ...
                randi([-48 48]);

            centerC = cc(k) + ...
                randi([-48 48]);

            [rows,cols] = ...
                getPatchCoordinates( ...
                    centerR, ...
                    centerC, ...
                    patchSize, ...
                    h, ...
                    w);

            Ipatch = I(rows,cols,:);

            Mpatch = MA(rows,cols);

            [predMask,~] = ...
                predictMAPatch( ...
                    net, ...
                    Ipatch, ...
                    threshold);

            %% Pixel statistics
            TPp = nnz( ...
                predMask & Mpatch);

            TNp = nnz( ...
                ~predMask & ~Mpatch);

            FPp = nnz( ...
                predMask & ~Mpatch);

            FNp = nnz( ...
                ~predMask & Mpatch);

            %% Global
            TP = TP + TPp;
            TN = TN + TNp;
            FP = FP + FPp;
            FN = FN + FNp;

            %% Image
            TPimg = TPimg + TPp;
            TNimg = TNimg + TNp;
            FPimg = FPimg + FPp;
            FNimg = FNimg + FNp;

            patchCount = patchCount + 1;

        end

    end

    %% ------------------------------------------------
    % Negative patches
    % -------------------------------------------------

    createdNeg = 0;

    attempts = 0;

    while ...
            createdNeg < negativePatchesPerImage && ...
            attempts < 1000

        attempts = attempts + 1;

        centerR = randi(h);

        centerC = randi(w);

        [rows,cols] = ...
            getPatchCoordinates( ...
                centerR, ...
                centerC, ...
                patchSize, ...
                h, ...
                w);

        retinaPatch = ...
            retinaMask(rows,cols);

        Mpatch = ...
            MA(rows,cols);

        %% Accept only normal retina
        if mean(retinaPatch(:)) > 0.70 && ...
                nnz(Mpatch) == 0

            Ipatch = ...
                I(rows,cols,:);

            [predMask,~] = ...
                predictMAPatch( ...
                    net, ...
                    Ipatch, ...
                    threshold);

            %% Pixel statistics
            TPp = nnz( ...
                predMask & Mpatch);

            TNp = nnz( ...
                ~predMask & ~Mpatch);

            FPp = nnz( ...
                predMask & ~Mpatch);

            FNp = nnz( ...
                ~predMask & Mpatch);

            %% Global
            TP = TP + TPp;
            TN = TN + TNp;
            FP = FP + FPp;
            FN = FN + FNp;

            %% Image
            TPimg = TPimg + TPp;
            TNimg = TNimg + TNp;
            FPimg = FPimg + FPp;
            FNimg = FNimg + FNp;

            createdNeg = createdNeg + 1;

            patchCount = patchCount + 1;

        end

    end

    %% ------------------------------------------------
    % Per-image metrics
    % -------------------------------------------------

    dice = ...
        (2 * TPimg) / ...
        max( ...
            2 * TPimg + FPimg + FNimg, ...
            eps);

    iou = ...
        TPimg / ...
        max( ...
            TPimg + FPimg + FNimg, ...
            eps);

    sensitivity = ...
        TPimg / ...
        max( ...
            TPimg + FNimg, ...
            eps);

    specificity = ...
        TNimg / ...
        max( ...
            TNimg + FPimg, ...
            eps);

    imageDice(end+1) = dice;
    imageIoU(end+1) = iou;
    imageSensitivity(end+1) = sensitivity;
    imageSpecificity(end+1) = specificity;

    imageTP(end+1) = TPimg;
    imageTN(end+1) = TNimg;
    imageFP(end+1) = FPimg;
    imageFN(end+1) = FNimg;

    %% ------------------------------------------------
    % Print
    % -------------------------------------------------

    fprintf( ...
        ['Image %s: Dice=%.4f, ' ...
         'IoU=%.4f, ' ...
         'Sensitivity=%.4f, ' ...
         'Specificity=%.4f\n'], ...
        imageID, ...
        dice, ...
        iou, ...
        sensitivity, ...
        specificity);

end

%% ---------------------------------------------------
% Mean metrics
% ----------------------------------------------------

meanDice = mean(imageDice);

meanIoU = mean(imageIoU);

meanSensitivity = ...
    mean(imageSensitivity);

meanSpecificity = ...
    mean(imageSpecificity);

%% ---------------------------------------------------
% Results
% ----------------------------------------------------

fprintf('\n');
fprintf('====================================================\n');
fprintf('MICROANEURYSM V3 VALIDATION COMPLETE\n');
fprintf('====================================================\n');

fprintf('\nValidation images : %d\n', ...
    numel(valMAIdx));

fprintf('Validation patches: %d\n', ...
    patchCount);

fprintf('\n');
fprintf('Mean Dice        : %.4f\n', ...
    meanDice);

fprintf('Mean IoU         : %.4f\n', ...
    meanIoU);

fprintf('Mean Sensitivity : %.4f\n', ...
    meanSensitivity);

fprintf('Mean Specificity : %.4f\n', ...
    meanSpecificity);

%% ---------------------------------------------------
% Pixel totals
% ----------------------------------------------------

fprintf('\nPixel-level totals:\n');

fprintf('TP = %d\n',TP);

fprintf('TN = %d\n',TN);

fprintf('FP = %d\n',FP);

fprintf('FN = %d\n',FN);

%% ---------------------------------------------------
% Compare with V1 and V2
% ----------------------------------------------------

fprintf('\n');
fprintf('====================================================\n');
fprintf('V1 / V2 / V3 COMPARISON\n');
fprintf('====================================================\n');

v1Dice = 0.5131;
v2Dice = 0.4326;

fprintf('\nV1 Dice: %.4f\n',v1Dice);

fprintf('V2 Dice: %.4f\n',v2Dice);

fprintf('V3 Dice: %.4f\n',meanDice);

fprintf('\nV3 - V1 Dice change: %+.4f\n', ...
    meanDice - v1Dice);

fprintf('V3 - V2 Dice change: %+.4f\n', ...
    meanDice - v2Dice);

%% ---------------------------------------------------
% Save results
% ----------------------------------------------------

results.microaneurysmV3 = struct( ...
    'meanDice',meanDice, ...
    'meanIoU',meanIoU, ...
    'meanSensitivity',meanSensitivity, ...
    'meanSpecificity',meanSpecificity, ...
    'TP',TP, ...
    'TN',TN, ...
    'FP',FP, ...
    'FN',FN, ...
    'threshold',threshold, ...
    'validationImages',numel(valMAIdx), ...
    'validationPatches',patchCount);

results.perImage.dice = ...
    imageDice;

results.perImage.iou = ...
    imageIoU;

results.perImage.sensitivity = ...
    imageSensitivity;

results.perImage.specificity = ...
    imageSpecificity;

results.perImage.TP = imageTP;
results.perImage.TN = imageTN;
results.perImage.FP = imageFP;
results.perImage.FN = imageFN;

if ~exist('results','dir')
    mkdir('results');
end

save( ...
    'results/microaneurysmValidationV3.mat', ...
    'results');

fprintf('\nSaved:\n');

fprintf('results/microaneurysmValidationV3.mat\n');

fprintf('====================================================\n');


%% ===================================================
% LOCAL FUNCTIONS
% ====================================================

function [rows,cols] = ...
    getPatchCoordinates( ...
        centerR, ...
        centerC, ...
        patchSize, ...
        h, ...
        w)

    r1 = round( ...
        centerR - patchSize/2);

    c1 = round( ...
        centerC - patchSize/2);

    r1 = max( ...
        1, ...
        min( ...
            r1, ...
            h-patchSize+1));

    c1 = max( ...
        1, ...
        min( ...
            c1, ...
            w-patchSize+1));

    rows = ...
        r1:r1+patchSize-1;

    cols = ...
        c1:c1+patchSize-1;

end


function [mask,scoreMap] = ...
    predictMAPatch( ...
        net, ...
        Ipatch, ...
        threshold)

    X = single(Ipatch);

    Y = predict( ...
        net, ...
        X);

    Y = extractdata(Y);

    Y = squeeze(Y);

    if ndims(Y) ~= 3 || ...
            size(Y,3) ~= 2

        error( ...
            'Unexpected Microaneurysm network output size.');

    end

    scoreMap = Y(:,:,2);

    mask = scoreMap >= threshold;

end