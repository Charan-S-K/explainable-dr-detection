%% SeeBeyond - Soft Exudate V2 Patch Model Validation
% Validation on held-out IDRiD images
% IMPORTANT: Network uses zerocenter normalization.
% Do NOT divide input images by 255.

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

fprintf('\n==============================================\n');
fprintf('SOFT EXUDATE V2 PATCH MODEL VALIDATION\n');
fprintf('==============================================\n');

%% Load model and validation split

S = load('models/softExudatePatchNetV2.mat', ...
    'softExudatePatchNetV2');

split = load('models/softExudateV2Split.mat', ...
    'valSEIdx');

net = S.softExudatePatchNetV2;
valSEIdx = split.valSEIdx;

fprintf('Validation images: %d\n', numel(valSEIdx));

%% Dataset paths

idridRoot = fullfile( ...
    pwd, 'datasets', 'IDRiD', 'Segmentation', 'A. Segmentation');

imageRoot = fullfile( ...
    idridRoot, '1. Original Images', 'a. Training Set');

seMaskRoot = fullfile( ...
    pwd, 'datasets', 'IDRiD', 'processed_masks', 'soft_exudates');

imds = imageDatastore(imageRoot);

%% Validation settings

patchSize = 256;

positivePerImage = 20;
randomNegPerImage = 20;

threshold = 0.20;

allDice = zeros(numel(valSEIdx),1);
allIoU = zeros(numel(valSEIdx),1);
allSensitivity = zeros(numel(valSEIdx),1);
allSpecificity = zeros(numel(valSEIdx),1);

totalTP = 0;
totalTN = 0;
totalFP = 0;
totalFN = 0;

totalPatches = 0;

%% Validate each held-out image

for n = 1:numel(valSEIdx)

    idx = valSEIdx(n);

    I = readimage(imds,idx);

    [~,imageID,~] = fileparts(imds.Files{idx});

    maskPath = fullfile( ...
        seMaskRoot, [imageID '.png']);

    if ~isfile(maskPath)
        error('Soft Exudate mask not found: %s', maskPath);
    end

    SE = imread(maskPath) > 0;

    [h,w,~] = size(I);

    [rr,cc] = find(SE);

    TP = 0;
    TN = 0;
    FP = 0;
    FN = 0;

    patchCount = 0;

    %% Positive patches

    if ~isempty(rr)

        for p = 1:positivePerImage

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
            Mpatch = SE(rows,cols);

            [predMask,~] = predictSEPatch(net,Ipatch,threshold);

            TP = TP + nnz(predMask & Mpatch);
            TN = TN + nnz(~predMask & ~Mpatch);
            FP = FP + nnz(predMask & ~Mpatch);
            FN = FN + nnz(~predMask & Mpatch);

            patchCount = patchCount + 1;
        end
    end

    %% Random negative patches

    createdNeg = 0;
    attemptsNeg = 0;

    while createdNeg < randomNegPerImage && attemptsNeg < 500

        attemptsNeg = attemptsNeg + 1;

        centerR = randi(h);
        centerC = randi(w);

        r1 = round(centerR - patchSize/2);
        c1 = round(centerC - patchSize/2);

        r1 = max(1,min(r1,h-patchSize+1));
        c1 = max(1,min(c1,w-patchSize+1));

        rows = r1:r1+patchSize-1;
        cols = c1:c1+patchSize-1;

        Mpatch = SE(rows,cols);

        % Require a completely negative patch.
        if nnz(Mpatch) == 0

            Ipatch = I(rows,cols,:);

[predMask,~] = predictSEPatch(net,Ipatch,threshold);

            TP = TP + nnz(predMask & Mpatch);
            TN = TN + nnz(~predMask & ~Mpatch);
            FP = FP + nnz(predMask & ~Mpatch);
            FN = FN + nnz(~predMask & Mpatch);

            createdNeg = createdNeg + 1;
            patchCount = patchCount + 1;
        end
    end

    %% Metrics

    dice = (2*TP) / max(2*TP + FP + FN,eps);

    iou = TP / max(TP + FP + FN,eps);

    sensitivity = TP / max(TP + FN,eps);

    specificity = TN / max(TN + FP,eps);

    allDice(n) = dice;
    allIoU(n) = iou;
    allSensitivity(n) = sensitivity;
    allSpecificity(n) = specificity;

    totalTP = totalTP + TP;
    totalTN = totalTN + TN;
    totalFP = totalFP + FP;
    totalFN = totalFN + FN;

    totalPatches = totalPatches + patchCount;

    fprintf(['Image %s: Dice=%.4f, IoU=%.4f, ' ...
             'Sensitivity=%.4f, Specificity=%.4f\n'], ...
             imageID,dice,iou,sensitivity,specificity);
end

%% Mean validation metrics

meanDice = mean(allDice);
meanIoU = mean(allIoU);
meanSensitivity = mean(allSensitivity);
meanSpecificity = mean(allSpecificity);

fprintf('\n==============================================\n');
fprintf('SOFT EXUDATE V2 VALIDATION COMPLETE\n');
fprintf('==============================================\n');

fprintf('Validation images : %d\n',numel(valSEIdx));
fprintf('Validation patches: %d\n\n',totalPatches);

fprintf('Mean Dice        : %.4f\n',meanDice);
fprintf('Mean IoU         : %.4f\n',meanIoU);
fprintf('Mean Sensitivity : %.4f\n',meanSensitivity);
fprintf('Mean Specificity : %.4f\n\n',meanSpecificity);

fprintf('Pixel-level totals:\n');
fprintf('TP = %d\n',totalTP);
fprintf('TN = %d\n',totalTN);
fprintf('FP = %d\n',totalFP);
fprintf('FN = %d\n',totalFN);

%% Save results

if ~exist('results','dir')
    mkdir('results');
end

save('results/softExudateV2Validation.mat', ...
    'meanDice', ...
    'meanIoU', ...
    'meanSensitivity', ...
    'meanSpecificity', ...
    'allDice', ...
    'allIoU', ...
    'allSensitivity', ...
    'allSpecificity', ...
    'totalTP', ...
    'totalTN', ...
    'totalFP', ...
    'totalFN', ...
    'valSEIdx', ...
    'totalPatches', ...
    'threshold');

fprintf('\nSaved:\n');
fprintf('results/softExudateV2Validation.mat\n');
fprintf('==============================================\n');


%% Local prediction function

function [mask,scoreMap] = predictSEPatch(net,Ipatch,threshold)

    % Convert uint8 image to single precision.
    X = single(Ipatch);

    % IMPORTANT:
    % The Soft Exudate V2 network uses an ImageInputLayer
    % with zerocenter normalization.
    %
    % Therefore DO NOT divide by 255.
    % The network input layer performs the normalization.

    Y = predict(net,X);

    Y = extractdata(Y);
    Y = squeeze(Y);

    % Expected output:
    % H x W x 2

    if ndims(Y) ~= 3 || size(Y,3) ~= 2
        error('Unexpected Soft Exudate network output size.');
    end

    % Class 2 = soft exudate.
    scoreMap = Y(:,:,2);

    % Threshold.
    mask = scoreMap >= threshold;
end