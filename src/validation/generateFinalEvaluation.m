function finalResults = generateFinalEvaluation()
% generateFinalEvaluation
% SeeBeyond Stage 5.3
% Creates the final unified evaluation summary.

fprintf('\n');
fprintf('============================================================\n');
fprintf('        SEEBEYOND FINAL UNIFIED EVALUATION\n');
fprintf('============================================================\n\n');

rootDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));

resultsDir = fullfile(rootDir,'results');

if ~exist(resultsDir,'dir')
    mkdir(resultsDir);
end

%% ============================================================
% STAGE 2 - SEGMENTATION
% =============================================================

fprintf('------------------------------------------------------------\n');
fprintf('STAGE 2 - RETINAL LESION SEGMENTATION\n');
fprintf('------------------------------------------------------------\n');

finalResults.stage2.name = "Retinal Lesion Segmentation";
finalResults.stage2.status = "Validated";
finalResults.stage2.dataset = "IDRiD";
finalResults.stage2.validationImages = 11;

% Full-image validation results

finalResults.stage2.hardExudate.dice = 0.18885;
finalResults.stage2.hardExudate.iou = 0.11275;
finalResults.stage2.hardExudate.sensitivity = 0.32612;
finalResults.stage2.hardExudate.specificity = 0.99366;
finalResults.stage2.hardExudate.precision = 0.21147;
finalResults.stage2.hardExudate.pixelAccuracy = 0.99122;

finalResults.stage2.haemorrhage.dice = 0.16088;
finalResults.stage2.haemorrhage.iou = 0.09354;
finalResults.stage2.haemorrhage.sensitivity = 0.4608;
finalResults.stage2.haemorrhage.specificity = 0.94992;
finalResults.stage2.haemorrhage.precision = 0.1332;
finalResults.stage2.haemorrhage.pixelAccuracy = 0.94552;

finalResults.stage2.microaneurysm.dice = 0.35787;
finalResults.stage2.microaneurysm.iou = 0.22310;
finalResults.stage2.microaneurysm.sensitivity = 0.43088;
finalResults.stage2.microaneurysm.specificity = 0.99915;
finalResults.stage2.microaneurysm.precision = 0.32725;
finalResults.stage2.microaneurysm.pixelAccuracy = 0.9985;

finalResults.stage2.softExudate.dice = 0.056798;
finalResults.stage2.softExudate.iou = 0.03329;
finalResults.stage2.softExudate.sensitivity = 0.2852;
finalResults.stage2.softExudate.specificity = 0.93366;
finalResults.stage2.softExudate.precision = 0.042346;
finalResults.stage2.softExudate.pixelAccuracy = 0.9327;

% Overall lesion performance

finalResults.stage2.overall.dice = 0.1911;
finalResults.stage2.overall.iou = 0.1157;
finalResults.stage2.overall.sensitivity = 0.3758;
finalResults.stage2.overall.specificity = 0.9691;
finalResults.stage2.overall.precision = 0.1786;
finalResults.stage2.overall.pixelAccuracy = 0.9670;

% Blood vessel validation was performed separately on DRIVE

finalResults.stage2.bloodVessel.dataset = "DRIVE";
finalResults.stage2.bloodVessel.dice = 0.5870;
finalResults.stage2.bloodVessel.iou = 0.4169;
finalResults.stage2.bloodVessel.sensitivity = 0.7645;
finalResults.stage2.bloodVessel.specificity = 0.9168;

fprintf('Dataset              : IDRiD\n');
fprintf('Validation images    : %d\n\n', ...
    finalResults.stage2.validationImages);

fprintf('Hard Exudate Dice    : %.2f%%\n', ...
    finalResults.stage2.hardExudate.dice * 100);

fprintf('Haemorrhage Dice     : %.2f%%\n', ...
    finalResults.stage2.haemorrhage.dice * 100);

fprintf('Microaneurysm Dice   : %.2f%%\n', ...
    finalResults.stage2.microaneurysm.dice * 100);

fprintf('Soft Exudate Dice    : %.2f%%\n', ...
    finalResults.stage2.softExudate.dice * 100);

fprintf('\nOverall Dice         : %.2f%%\n', ...
    finalResults.stage2.overall.dice * 100);

fprintf('Overall IoU          : %.2f%%\n', ...
    finalResults.stage2.overall.iou * 100);

fprintf('Overall Sensitivity  : %.2f%%\n', ...
    finalResults.stage2.overall.sensitivity * 100);

fprintf('Overall Specificity  : %.2f%%\n', ...
    finalResults.stage2.overall.specificity * 100);

fprintf('Overall Precision    : %.2f%%\n', ...
    finalResults.stage2.overall.precision * 100);

fprintf('Overall Pixel Acc.   : %.2f%%\n', ...
    finalResults.stage2.overall.pixelAccuracy * 100);

fprintf('\nBlood Vessel - DRIVE\n');
fprintf('Dice                 : %.2f%%\n', ...
    finalResults.stage2.bloodVessel.dice * 100);

fprintf('IoU                  : %.2f%%\n', ...
    finalResults.stage2.bloodVessel.iou * 100);

fprintf('Sensitivity          : %.2f%%\n', ...
    finalResults.stage2.bloodVessel.sensitivity * 100);

fprintf('Specificity          : %.2f%%\n', ...
    finalResults.stage2.bloodVessel.specificity * 100);

%% ============================================================
% STAGE 3 - DR CLASSIFICATION
% =============================================================

fprintf('\n');
fprintf('------------------------------------------------------------\n');
fprintf('STAGE 3 - DIABETIC RETINOPATHY CLASSIFICATION\n');
fprintf('------------------------------------------------------------\n');

finalResults.stage3.name = ...
    "Diabetic Retinopathy Classification";

finalResults.stage3.status = "Validated";

finalResults.stage3.dataset = "APTOS2019";

finalResults.stage3.testImages = 554;

finalResults.stage3.testAccuracy = 0.8195;
finalResults.stage3.macroPrecision = 0.6948;
finalResults.stage3.macroRecall = 0.6159;
finalResults.stage3.macroF1 = 0.6378;

% Per-class F1 scores

finalResults.stage3.classF1.No_DR = 0.97398;
finalResults.stage3.classF1.Mild = 0.58586;
finalResults.stage3.classF1.Moderate = 0.76791;
finalResults.stage3.classF1.Severe = 0.27907;
finalResults.stage3.classF1.Proliferative_DR = 0.58228;

% Severe recall

finalResults.stage3.severeRecall = 0.2000;

fprintf('Dataset              : %s\n', ...
    finalResults.stage3.dataset);

fprintf('Test images          : %d\n', ...
    finalResults.stage3.testImages);

fprintf('Test Accuracy        : %.2f%%\n', ...
    finalResults.stage3.testAccuracy * 100);

fprintf('Macro Precision      : %.2f%%\n', ...
    finalResults.stage3.macroPrecision * 100);

fprintf('Macro Recall         : %.2f%%\n', ...
    finalResults.stage3.macroRecall * 100);

fprintf('Macro F1             : %.2f%%\n', ...
    finalResults.stage3.macroF1 * 100);

fprintf('Severe DR Recall     : %.2f%%\n', ...
    finalResults.stage3.severeRecall * 100);

%% ============================================================
% STAGE 1
% =============================================================

fprintf('\n');
fprintf('------------------------------------------------------------\n');
fprintf('STAGE 1 - IMAGE QUALITY & PREPROCESSING\n');
fprintf('------------------------------------------------------------\n');

finalResults.stage1.name = ...
    "Image Quality Assessment and Preprocessing";

finalResults.stage1.status = "Complete";

finalResults.stage1.components = [ ...
    "Blur detection"
    "Illumination assessment"
    "Field-of-view evaluation"
    "Fundus image enhancement"
    ];

fprintf('Status               : COMPLETE\n');
fprintf('Components           : Blur + Illumination + FOV + Enhancement\n');

%% ============================================================
% STAGE 4
% =============================================================

fprintf('\n');
fprintf('------------------------------------------------------------\n');
fprintf('STAGE 4 - END-TO-END INTEGRATION\n');
fprintf('------------------------------------------------------------\n');

finalResults.stage4.name = ...
    "SeeBeyond End-to-End Integration";

finalResults.stage4.status = "Complete";

finalResults.stage4.pipeline = ...
    "Quality -> Enhancement -> Classification -> Segmentation -> Report";

fprintf('Status               : COMPLETE\n');
fprintf('Pipeline              : %s\n', ...
    finalResults.stage4.pipeline);

%% ============================================================
% FINAL STATUS
% =============================================================

finalResults.project = "SeeBeyond";

finalResults.stage5.name = "Validation and Evaluation";

finalResults.stage5.status = "Complete";

finalResults.stage5.stage51 = ...
    "DR classifier validation - COMPLETE";

finalResults.stage5.stage52 = ...
    "Full-image segmentation validation - COMPLETE";

finalResults.stage5.stage53 = ...
    "Unified evaluation summary - COMPLETE";

%% ============================================================
% SAVE RESULTS
% =============================================================

outputFile = fullfile( ...
    resultsDir, ...
    'finalSeeBeyondEvaluation.mat');

save(outputFile,'finalResults');

fprintf('\n');
fprintf('============================================================\n');
fprintf('FINAL SEEBEYOND EVALUATION\n');
fprintf('============================================================\n');

fprintf('\nStage 1 : COMPLETE\n');
fprintf('Stage 2 : COMPLETE\n');
fprintf('Stage 3 : COMPLETE\n');
fprintf('Stage 4 : COMPLETE\n');
fprintf('Stage 5 : COMPLETE\n');

fprintf('\nSaved file:\n');
fprintf('%s\n',outputFile);

fprintf('\n============================================================\n');
fprintf('STAGE 5.3 COMPLETE\n');
fprintf('============================================================\n\n');

end