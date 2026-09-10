%% SeeBeyond - Haemorrhage V4 Final Validation

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

fprintf('\n========================================\n');
fprintf('HAEMORRHAGE V4 FINAL VALIDATION\n');
fprintf('========================================\n');

%% Load V4

load('models/haemorrhagePatchNetV4.mat', ...
    'haemorrhagePatchNetV4','valHEIdx');

%% Dataset paths

idridRoot = fullfile(pwd, ...
    'datasets','IDRiD','Segmentation','A. Segmentation');

imageRoot = fullfile(idridRoot, ...
    '1. Original Images','a. Training Set');

heMaskRoot = fullfile(pwd, ...
    'datasets','IDRiD','processed_masks','haemorrhages');

imdsHE = imageDatastore(imageRoot);
imdsHEVal = subset(imdsHE,valHEIdx);

numVal = numel(imdsHEVal.Files);

fprintf('Validation images : %d\n',numVal);

%% Threshold sweep

thresholds = [ ...
    0.05 0.10 0.15 0.20 0.25 ...
    0.30 0.35 0.40 0.45 0.50 ...
    0.55 0.60 0.65 0.70 0.75 ...
    0.80 0.85 0.90 0.95];

numT = numel(thresholds);

diceByT = nan(numVal,numT);
iouByT  = nan(numVal,numT);
sensByT = nan(numVal,numT);
specByT = nan(numVal,numT);

fpByT = zeros(numVal,numT);
fnByT = zeros(numVal,numT);

%% Run each image once

for i = 1:numVal

    I = readimage(imdsHEVal,i);

    [~,imageID,~] = ...
        fileparts(imdsHEVal.Files{i});

    GT = imread( ...
        fullfile(heMaskRoot,[imageID '.png'])) > 0;

    fprintf('\n[%d/%d] %s\n', ...
        i,numVal,imageID);

    prob = predictHaemorrhageProbMapV2( ...
        I,haemorrhagePatchNetV4);

    fprintf('Probability range : %.6f - %.6f\n', ...
        min(prob(:)),max(prob(:)));

    for t = 1:numT

        Pred = prob >= thresholds(t);

        Pred = bwareaopen(Pred,6);

        TP = sum(Pred(:) & GT(:));
        TN = sum(~Pred(:) & ~GT(:));
        FP = sum(Pred(:) & ~GT(:));
        FN = sum(~Pred(:) & GT(:));

        fpByT(i,t) = FP;
        fnByT(i,t) = FN;

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

%% Mean metrics

meanDice = mean(diceByT,1,'omitnan');
meanIoU  = mean(iouByT,1,'omitnan');
meanSens = mean(sensByT,1,'omitnan');
meanSpec = mean(specByT,1,'omitnan');

meanFP = mean(fpByT,1);
meanFN = mean(fnByT,1);

%% Results table

resultsV4 = table( ...
    thresholds', ...
    meanDice', ...
    meanIoU', ...
    meanSens', ...
    meanSpec', ...
    meanFP', ...
    meanFN', ...
    'VariableNames', ...
    {'Threshold','MeanDice','MeanIoU', ...
     'MeanSensitivity','MeanSpecificity', ...
     'MeanFP','MeanFN'});

disp(resultsV4)

%% Best threshold

[bestDiceHEV4,bestIdx] = max(meanDice);

bestThresholdHEV4 = thresholds(bestIdx);
bestIoUHEV4 = meanIoU(bestIdx);
bestSensitivityHEV4 = meanSens(bestIdx);
bestSpecificityHEV4 = meanSpec(bestIdx);

bestMeanFPHEV4 = meanFP(bestIdx);
bestMeanFNHEV4 = meanFN(bestIdx);

%% Compare with original baseline

baselineDice = 0.0248;

diceImprovement = ...
    bestDiceHEV4 - baselineDice;

fprintf('\n========================================\n');
fprintf('BEST HAEMORRHAGE V4 RESULT\n');
fprintf('========================================\n');

fprintf('Threshold        : %.2f\n', ...
    bestThresholdHEV4);

fprintf('Mean Dice        : %.4f\n', ...
    bestDiceHEV4);

fprintf('Mean IoU         : %.4f\n', ...
    bestIoUHEV4);

fprintf('Mean Sensitivity : %.4f\n', ...
    bestSensitivityHEV4);

fprintf('Mean Specificity : %.4f\n', ...
    bestSpecificityHEV4);

fprintf('Mean FP           : %.1f\n', ...
    bestMeanFPHEV4);

fprintf('Mean FN           : %.1f\n', ...
    bestMeanFNHEV4);

fprintf('\nOriginal Dice    : %.4f\n',baselineDice);

fprintf('Improvement      : %+.4f\n', ...
    diceImprovement);

if bestDiceHEV4 > baselineDice

    fprintf('\nV4 BEATS ORIGINAL MODEL ✅\n');

else

    fprintf('\nV4 DOES NOT BEAT ORIGINAL MODEL ❌\n');

end

%% Save

save('results/haemorrhageV4Validation.mat', ...
    'resultsV4', ...
    'bestThresholdHEV4', ...
    'bestDiceHEV4', ...
    'bestIoUHEV4', ...
    'bestSensitivityHEV4', ...
    'bestSpecificityHEV4', ...
    'bestMeanFPHEV4', ...
    'bestMeanFNHEV4', ...
    'diceImprovement');

fprintf('\nSaved: results/haemorrhageV4Validation.mat\n');