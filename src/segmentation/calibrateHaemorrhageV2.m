%% SeeBeyond - Haemorrhage V2 Validation
% Compare V2 against old baseline Dice = 0.0248

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

fprintf('\n========================================\n');
fprintf('HAEMORRHAGE V2 VALIDATION\n');
fprintf('========================================\n');

%% Load V2 model

load('models/haemorrhagePatchNetV2.mat', ...
    'haemorrhagePatchNetV2', ...
    'valHEIdx');

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

%% Thresholds

thresholds = ...
    [0.05 0.10:0.05:0.90 0.95];

numT = numel(thresholds);

diceByT = nan(numVal,numT);
iouByT  = nan(numVal,numT);
sensByT = nan(numVal,numT);
specByT = nan(numVal,numT);

FPByT = zeros(numVal,numT);
FNByT = zeros(numVal,numT);

%% =========================================================
% Run network ONCE per image
%% =========================================================

for i = 1:numVal

    I = readimage(imdsHEVal,i);

    [~,imageID,~] = ...
        fileparts(imdsHEVal.Files{i});

    GT = imread( ...
        fullfile(heMaskRoot,[imageID '.png'])) > 0;

    fprintf('\n[%d/%d] Predicting %s\n', ...
        i,numVal,imageID);

    prob = predictHaemorrhageProbMapV2( ...
        I,haemorrhagePatchNetV2);

    fprintf('Probability range : %.4f - %.4f\n', ...
        min(prob(:)),max(prob(:)));

    %% Test thresholds

    for t = 1:numT

        Pred = prob >= thresholds(t);

        % Remove only very small isolated noise
        Pred = bwareaopen(Pred,6);

        TP = sum(Pred(:) & GT(:));
        TN = sum(~Pred(:) & ~GT(:));
        FP = sum(Pred(:) & ~GT(:));
        FN = sum(~Pred(:) & GT(:));

        FPByT(i,t) = FP;
        FNByT(i,t) = FN;

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

%% =========================================================
% Mean metrics
%% =========================================================

meanDice = mean(diceByT,1,'omitnan');
meanIoU  = mean(iouByT,1,'omitnan');
meanSens = mean(sensByT,1,'omitnan');
meanSpec = mean(specByT,1,'omitnan');

meanFP = mean(FPByT,1);
meanFN = mean(FNByT,1);

thresholdTable = table( ...
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

disp(thresholdTable)

%% Best Dice threshold

[bestDiceHEV2,bestIdx] = max(meanDice);

bestThresholdHEV2 = thresholds(bestIdx);
bestIoUHEV2 = meanIoU(bestIdx);
bestSensitivityHEV2 = meanSens(bestIdx);
bestSpecificityHEV2 = meanSpec(bestIdx);
bestMeanFPHEV2 = meanFP(bestIdx);
bestMeanFNHEV2 = meanFN(bestIdx);

fprintf('\n========================================\n');
fprintf('BEST HAEMORRHAGE V2 RESULT\n');
fprintf('========================================\n');

fprintf('Threshold        : %.2f\n', ...
    bestThresholdHEV2);

fprintf('Mean Dice        : %.4f\n', ...
    bestDiceHEV2);

fprintf('Mean IoU         : %.4f\n', ...
    bestIoUHEV2);

fprintf('Mean Sensitivity : %.4f\n', ...
    bestSensitivityHEV2);

fprintf('Mean Specificity : %.4f\n', ...
    bestSpecificityHEV2);

fprintf('Mean FP           : %.1f\n', ...
    bestMeanFPHEV2);

fprintf('Mean FN           : %.1f\n', ...
    bestMeanFNHEV2);

fprintf('\nOLD BASELINE DICE : 0.0248\n');

improvement = ...
    bestDiceHEV2 - 0.0248;

fprintf('Dice improvement : %+.4f\n', ...
    improvement);

if bestDiceHEV2 > 0.0248
    fprintf('\nRESULT: V2 IS BETTER ✅\n');
else
    fprintf('\nRESULT: V2 DID NOT IMPROVE ❌\n');
end

%% Save validation

save('results/haemorrhageV2Validation.mat', ...
    'thresholdTable', ...
    'bestThresholdHEV2', ...
    'bestDiceHEV2', ...
    'bestIoUHEV2', ...
    'bestSensitivityHEV2', ...
    'bestSpecificityHEV2', ...
    'bestMeanFPHEV2', ...
    'bestMeanFNHEV2');

fprintf('\nValidation results saved.\n');