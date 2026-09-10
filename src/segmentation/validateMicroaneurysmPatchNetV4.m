%% SeeBeyond - Microaneurysm V4 Validation
% Uses the same 11 held-out images and same 440-patch protocol
% used for V1, V2 and V3.

clear;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

fprintf('\n');
fprintf('====================================================\n');
fprintf('MICROANEURYSM V4 PATCH MODEL VALIDATION\n');
fprintf('====================================================\n');

%% Load model

modelFile = 'models/microaneurysmPatchNetV4.mat';

if ~isfile(modelFile)
    error('V4 model not found: %s',modelFile);
end

S = load(modelFile,'microaneurysmPatchNetV4');
net = S.microaneurysmPatchNetV4;

%% Validation split

splitFile = 'models/microaneurysmV2Split.mat';

if ~isfile(splitFile)
    error('Validation split not found: %s',splitFile);
end

Ssplit = load(splitFile,'valMAIdx');
valMAIdx = Ssplit.valMAIdx;

fprintf('Validation images: %d\n',numel(valMAIdx));

%% Paths

idridRoot = fullfile(pwd, ...
    'datasets','IDRiD','Segmentation','A. Segmentation');

imageRoot = fullfile(idridRoot, ...
    '1. Original Images','a. Training Set');

maMaskRoot = fullfile(pwd, ...
    'datasets','IDRiD','processed_masks','microaneurysms');

imds = imageDatastore(imageRoot);

%% Validation settings

patchSize = 256;
positivePatchesPerImage = 20;
negativePatchesPerImage = 20;

threshold = 0.50;

rng(42);

fprintf('\nValidation settings:\n');
fprintf('Patch size       : 256 x 256\n');
fprintf('Positive patches : 20 / image\n');
fprintf('Negative patches : 20 / image\n');
fprintf('Threshold        : %.2f\n',threshold);

%% Statistics

TP = 0;
TN = 0;
FP = 0;
FN = 0;

patchCount = 0;

imageDice = [];
imageIoU = [];
imageSensitivity = [];
imageSpecificity = [];

%% Validation loop

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

    %% Retina mask

    G = im2double(rgb2gray(I));

    retinaMask = G > 0.03;

    retinaMask = bwareafilt(retinaMask,1);

    retinaMask = imfill(retinaMask,'holes');

    %% Per-image statistics

    TPimg = 0;
    TNimg = 0;
    FPimg = 0;
    FNimg = 0;

    %% Positive patches

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

            [predMask,~] = predictMAPatchV4( ...
                net,Ipatch,threshold);

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

    %% Negative patches

    createdNeg = 0;
    attempts = 0;

    while createdNeg < negativePatchesPerImage && attempts < 1000

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

        if mean(retinaPatch(:)) > 0.70 && nnz(Mpatch) == 0

            Ipatch = I(rows,cols,:);

            [predMask,~] = predictMAPatchV4( ...
                net,Ipatch,threshold);

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

    %% Metrics

    dice = (2*TPimg) / ...
        max(2*TPimg + FPimg + FNimg,eps);

    iou = TPimg / ...
        max(TPimg + FPimg + FNimg,eps);

    sensitivity = TPimg / ...
        max(TPimg + FNimg,eps);

    specificity = TNimg / ...
        max(TNimg + FPimg,eps);

    imageDice(end+1) = dice;
    imageIoU(end+1) = iou;
    imageSensitivity(end+1) = sensitivity;
    imageSpecificity(end+1) = specificity;

    fprintf(['Image %s: Dice=%.4f, IoU=%.4f, ' ...
        'Sensitivity=%.4f, Specificity=%.4f\n'], ...
        imageID,dice,iou,sensitivity,specificity);
end

%% Mean metrics

meanDice = mean(imageDice);
meanIoU = mean(imageIoU);
meanSensitivity = mean(imageSensitivity);
meanSpecificity = mean(imageSpecificity);

%% Display

fprintf('\n');
fprintf('====================================================\n');
fprintf('MICROANEURYSM V4 VALIDATION COMPLETE\n');
fprintf('====================================================\n');

fprintf('\nValidation images : %d\n',numel(valMAIdx));
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

%% Comparison

fprintf('\n');
fprintf('====================================================\n');
fprintf('V1 / V2 / V3 / V4 COMPARISON\n');
fprintf('====================================================\n');

fprintf('V1 Dice: 0.5131\n');
fprintf('V2 Dice: 0.4326\n');
fprintf('V3 Dice: 0.4890\n');
fprintf('V4 Dice: %.4f\n',meanDice);

fprintf('\nV4 - V1 Dice change: %+.4f\n',meanDice-0.5131);
fprintf('V4 - V2 Dice change: %+.4f\n',meanDice-0.4326);
fprintf('V4 - V3 Dice change: %+.4f\n',meanDice-0.4890);

%% Save results

if ~exist('results','dir')
    mkdir('results');
end

results.microaneurysm = struct( ...
    'model','Microaneurysm_V4', ...
    'meanDice',meanDice, ...
    'meanIoU',meanIoU, ...
    'meanSensitivity',meanSensitivity, ...
    'meanSpecificity',meanSpecificity, ...
    'TP',TP, ...
    'TN',TN, ...
    'FP',FP, ...
    'FN',FN, ...
    'validationImages',numel(valMAIdx), ...
    'validationPatches',patchCount, ...
    'threshold',threshold);

results.perImage.dice = imageDice;
results.perImage.iou = imageIoU;
results.perImage.sensitivity = imageSensitivity;
results.perImage.specificity = imageSpecificity;

save('results/microaneurysmValidationV4.mat','results');

fprintf('\nSaved:\n');
fprintf('results/microaneurysmValidationV4.mat\n');

fprintf('====================================================\n');


%% Local prediction function

function [mask,scoreMap] = predictMAPatchV4(net,Ipatch,threshold)

    X = single(Ipatch);

    % The network uses its own input normalization.
    Y = predict(net,X);

    if isa(Y,'dlarray')
        Y = extractdata(Y);
    end

    if isa(Y,'gpuArray')
        Y = gather(Y);
    end

    Y = squeeze(Y);

    if ndims(Y) ~= 3 || size(Y,3) ~= 2
        error('Unexpected Microaneurysm V4 network output size.');
    end

    scoreMap = Y(:,:,2);

    mask = scoreMap >= threshold;

end