%% SeeBeyond - Soft Exudate Threshold Calibration

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

fprintf('\n===== SOFT EXUDATE THRESHOLD CALIBRATION =====\n');

%% Load model

load('models/softExudatePatchNet.mat', ...
    'softExudatePatchNet','valSEIdx');

%% Paths

idridRoot = fullfile(pwd, ...
    'datasets','IDRiD','Segmentation','A. Segmentation');

imageRoot = fullfile(idridRoot, ...
    '1. Original Images','a. Training Set');

seMaskRoot = fullfile(pwd, ...
    'datasets','IDRiD','processed_masks','soft_exudates');

imdsSE = imageDatastore(imageRoot);
imdsSEVal = subset(imdsSE,valSEIdx);

%% Thresholds to test

thresholds = [ ...
    0.01 0.02 0.05 ...
    0.10:0.05:0.90];

numT = numel(thresholds);
numImages = numel(imdsSEVal.Files);

diceByT = nan(numImages,numT);
iouByT  = nan(numImages,numT);
sensByT = nan(numImages,numT);
specByT = nan(numImages,numT);

fprintf('Validation images : %d\n',numImages);
fprintf('Thresholds tested : %d\n\n',numT);

%% =====================================================
% Predict each retina ONCE
%% =====================================================

for i = 1:numImages

    I = readimage(imdsSEVal,i);

    [~,imageID,~] = fileparts(imdsSEVal.Files{i});

    GT = imread( ...
        fullfile(seMaskRoot,[imageID '.png'])) > 0;

    fprintf('Predicting %d/%d : %s\n', ...
        i,numImages,imageID);

    prob = predictSoftExudateProbMap( ...
        I,softExudatePatchNet);

    fprintf('   Probability range: %.4f - %.4f\n', ...
        min(prob(:)),max(prob(:)));

    %% Test all thresholds without running network again

    for t = 1:numT

        Pred = prob >= thresholds(t);

        % Remove only tiny isolated noise
        Pred = bwareaopen(Pred,3);

        TP = sum(Pred(:) & GT(:));
        TN = sum(~Pred(:) & ~GT(:));
        FP = sum(Pred(:) & ~GT(:));
        FN = sum(~Pred(:) & GT(:));

        if any(GT(:))

            diceByT(i,t) = ...
                (2*TP)/(2*TP + FP + FN + eps);

            iouByT(i,t) = ...
                TP/(TP + FP + FN + eps);

            sensByT(i,t) = ...
                TP/(TP + FN + eps);
        end

        specByT(i,t) = ...
            TN/(TN + FP + eps);
    end
end

%% Mean results

meanDiceByT = mean(diceByT,1,'omitnan');
meanIoUByT  = mean(iouByT,1,'omitnan');
meanSensByT = mean(sensByT,1,'omitnan');
meanSpecByT = mean(specByT,1,'omitnan');

%% Table

thresholdResults = table( ...
    thresholds', ...
    meanDiceByT', ...
    meanIoUByT', ...
    meanSensByT', ...
    meanSpecByT', ...
    'VariableNames', ...
    {'Threshold','MeanDice','MeanIoU', ...
     'MeanSensitivity','MeanSpecificity'});

disp(thresholdResults)

%% Find best Dice threshold

[bestDiceSE,bestIdx] = max(meanDiceByT);

bestThresholdSE = thresholds(bestIdx);
bestIoUSE = meanIoUByT(bestIdx);
bestSensitivitySE = meanSensByT(bestIdx);
bestSpecificitySE = meanSpecByT(bestIdx);

fprintf('\n========================================\n');
fprintf('BEST SOFT EXUDATE THRESHOLD\n');
fprintf('========================================\n');

fprintf('Threshold        : %.2f\n',bestThresholdSE);
fprintf('Mean Dice        : %.4f\n',bestDiceSE);
fprintf('Mean IoU         : %.4f\n',bestIoUSE);
fprintf('Mean Sensitivity : %.4f\n',bestSensitivitySE);
fprintf('Mean Specificity : %.4f\n',bestSpecificitySE);

fprintf('Positive images  : %d / %d\n', ...
    sum(any(~isnan(diceByT),2)),numImages);

%% Save calibration result

save('results/softExudateThresholdCalibration.mat', ...
    'thresholdResults', ...
    'bestThresholdSE', ...
    'bestDiceSE', ...
    'bestIoUSE', ...
    'bestSensitivitySE', ...
    'bestSpecificitySE', ...
    'diceByT', ...
    'iouByT', ...
    'sensByT', ...
    'specByT');

fprintf('\nSaved calibration results.\n');