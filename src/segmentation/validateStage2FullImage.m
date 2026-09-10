function validateStage2FullImage()
% SeeBeyond - Stage 2 Full-Image Validation
%
% Full-image validation on the 11 held-out IDRiD images.
%
% Metrics:
% Dice, IoU, Sensitivity, Specificity,
% Precision and Pixel Accuracy.
%
% Blood vessels are NOT evaluated on IDRiD because
% IDRiD does not provide vessel ground truth.

clc;

fprintf('\n');
fprintf('============================================================\n');
fprintf('SEEBEYOND STAGE-2 FULL-IMAGE VALIDATION\n');
fprintf('============================================================\n');

%% Paths

projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));

imageRoot = fullfile( ...
    projectRoot, ...
    'datasets','IDRiD','Segmentation', ...
    'A. Segmentation', ...
    '1. Original Images','a. Training Set');

gtRoot = fullfile( ...
    projectRoot, ...
    'datasets','IDRiD','Segmentation', ...
    'A. Segmentation', ...
    '2. All Segmentation Groundtruths','a. Training Set');
softGtRoot = fullfile( ...
    projectRoot, ...
    'datasets','IDRiD','processed_masks', ...
    'soft_exudates');

%% Validation images

valIDs = [36 13 8 54 44 53 34 2 35 51 12];

nImages = numel(valIDs);

fprintf('Validation images : %d\n',nImages);

%% Structures

structureNames = [
    "Blood Vessel"
    "Hard Exudate"
    "Haemorrhage"
    "Microaneurysm"
    "Soft Exudate"
    ];

nStructures = 5;

%% Storage

allDice = NaN(nImages,nStructures);
allIoU = NaN(nImages,nStructures);
allSensitivity = NaN(nImages,nStructures);
allSpecificity = NaN(nImages,nStructures);
allPrecision = NaN(nImages,nStructures);
allAccuracy = NaN(nImages,nStructures);

allTP = NaN(nImages,nStructures);
allTN = NaN(nImages,nStructures);
allFP = NaN(nImages,nStructures);
allFN = NaN(nImages,nStructures);

%% Process images

for k = 1:nImages

    id = valIDs(k);

    imageID = sprintf('IDRiD_%02d',id);

    fprintf('\n--------------------------------------------\n');
    fprintf('%s (%d/%d)\n',imageID,k,nImages);
    fprintf('--------------------------------------------\n');

    %% Load original image

    imagePath = fullfile( ...
        imageRoot, ...
        [imageID '.jpg']);

    if ~isfile(imagePath)

        error( ...
            'Image not found: %s', ...
            imagePath);

    end

    I = imread(imagePath);

    %% Run complete Stage-2 pipeline

    R = runSegmentationPipeline(I);

    predictedMasks = {
        R.vesselMask
        R.hardExudateMask
        R.haemorrhageMask
        R.microaneurysmMask
        R.softExudateMask
        };

    %% Ground truth paths

    hardExudatePath = fullfile( ...
        gtRoot, ...
        '3. Hard Exudates', ...
        [imageID '_EX.tif']);

    haemorrhagePath = fullfile( ...
        gtRoot, ...
        '2. Haemorrhages', ...
        [imageID '_HE.tif']);

    microaneurysmPath = fullfile( ...
        gtRoot, ...
        '1. Microaneurysms', ...
        [imageID '_MA.tif']);

    softExudatePath = fullfile( ...
        softGtRoot, ...
        [imageID '.png']);

    %% Load GT

    gtMasks = cell(5,1);

    % No IDRiD vessel GT
    gtMasks{1} = [];

    gtMasks{2} = loadMask( ...
        hardExudatePath, ...
        size(I));

    gtMasks{3} = loadMask( ...
        haemorrhagePath, ...
        size(I));

    gtMasks{4} = loadMask( ...
        microaneurysmPath, ...
        size(I));

    gtMasks{5} = loadMask( ...
        softExudatePath, ...
        size(I));

    %% Calculate metrics

    for s = 1:nStructures

        pred = predictedMasks{s};
        gt = gtMasks{s};

        if isempty(gt)

            continue;

        end

        [dice,iou,sensitivity,specificity, ...
            precision,accuracy,TP,TN,FP,FN] = ...
            calculateMetrics(pred,gt);

        allDice(k,s) = dice;
        allIoU(k,s) = iou;
        allSensitivity(k,s) = sensitivity;
        allSpecificity(k,s) = specificity;
        allPrecision(k,s) = precision;
        allAccuracy(k,s) = accuracy;

        allTP(k,s) = TP;
        allTN(k,s) = TN;
        allFP(k,s) = FP;
        allFN(k,s) = FN;

        fprintf( ...
            '%-18s Dice=%0.4f  IoU=%0.4f  Sens=%0.4f  Spec=%0.4f  Acc=%0.4f\n', ...
            structureNames(s), ...
            dice, ...
            iou, ...
            sensitivity, ...
            specificity, ...
            accuracy);

    end

end

%% Mean metrics

meanDice = mean(allDice,1,'omitnan');
meanIoU = mean(allIoU,1,'omitnan');
meanSensitivity = mean(allSensitivity,1,'omitnan');
meanSpecificity = mean(allSpecificity,1,'omitnan');
meanPrecision = mean(allPrecision,1,'omitnan');
meanAccuracy = mean(allAccuracy,1,'omitnan');

%% Results table

resultsTable = table( ...
    structureNames, ...
    meanDice', ...
    meanIoU', ...
    meanSensitivity', ...
    meanSpecificity', ...
    meanPrecision', ...
    meanAccuracy', ...
    'VariableNames',{ ...
    'Structure', ...
    'Dice', ...
    'IoU', ...
    'Sensitivity', ...
    'Specificity', ...
    'Precision', ...
    'PixelAccuracy'});

%% Display final results

fprintf('\n');
fprintf('============================================================\n');
fprintf('FINAL STAGE-2 FULL-IMAGE RESULTS\n');
fprintf('============================================================\n');

disp(resultsTable);

%% Overall lesion average

lesionDice = mean(meanDice(2:5),'omitnan');
lesionIoU = mean(meanIoU(2:5),'omitnan');
lesionSensitivity = mean(meanSensitivity(2:5),'omitnan');
lesionSpecificity = mean(meanSpecificity(2:5),'omitnan');
lesionPrecision = mean(meanPrecision(2:5),'omitnan');
lesionAccuracy = mean(meanAccuracy(2:5),'omitnan');

fprintf('\n');
fprintf('============================================================\n');
fprintf('OVERALL LESION PERFORMANCE\n');
fprintf('============================================================\n');

fprintf('Mean Dice        : %.4f\n',lesionDice);
fprintf('Mean IoU         : %.4f\n',lesionIoU);
fprintf('Mean Sensitivity : %.4f\n',lesionSensitivity);
fprintf('Mean Specificity : %.4f\n',lesionSpecificity);
fprintf('Mean Precision   : %.4f\n',lesionPrecision);
fprintf('Mean Pixel Acc.  : %.4f\n',lesionAccuracy);

%% Save

save( ...
    fullfile(projectRoot, ...
    'results','stage2FullImageValidation.mat'), ...
    'resultsTable', ...
    'allDice', ...
    'allIoU', ...
    'allSensitivity', ...
    'allSpecificity', ...
    'allPrecision', ...
    'allAccuracy', ...
    'allTP', ...
    'allTN', ...
    'allFP', ...
    'allFN', ...
    'valIDs', ...
    'lesionDice', ...
    'lesionIoU', ...
    'lesionSensitivity', ...
    'lesionSpecificity', ...
    'lesionPrecision', ...
    'lesionAccuracy');

fprintf('\nSaved:\n');
fprintf('results/stage2FullImageValidation.mat\n');

fprintf('\n');
fprintf('============================================================\n');
fprintf('STAGE-2 FULL-IMAGE VALIDATION COMPLETE\n');
fprintf('============================================================\n');

end


%% =========================================================
% Load mask
%% =========================================================

function mask = loadMask(path,targetSize)

if ~isfile(path)

    error( ...
        'Ground-truth mask not found: %s', ...
        path);

end

M = imread(path);

if ndims(M) > 2
    M = M(:,:,1);
end

M = imresize( ...
    M, ...
    targetSize(1:2), ...
    'nearest');

mask = M > 0;

end


%% =========================================================
% Metrics
%% =========================================================

function [dice,iou,sensitivity,specificity, ...
    precision,accuracy,TP,TN,FP,FN] = ...
    calculateMetrics(pred,gt)

pred = logical(pred);
gt = logical(gt);

TP = nnz(pred & gt);
TN = nnz(~pred & ~gt);
FP = nnz(pred & ~gt);
FN = nnz(~pred & gt);

epsilon = 1e-12;

dice = ...
    (2*TP) / ...
    (2*TP + FP + FN + epsilon);

iou = ...
    TP / ...
    (TP + FP + FN + epsilon);

sensitivity = ...
    TP / ...
    (TP + FN + epsilon);

specificity = ...
    TN / ...
    (TN + FP + epsilon);

precision = ...
    TP / ...
    (TP + FP + epsilon);

accuracy = ...
    (TP + TN) / ...
    (TP + TN + FP + FN + epsilon);

end