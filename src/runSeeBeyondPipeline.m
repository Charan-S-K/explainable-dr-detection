function results = runSeeBeyondPipeline(inputImage)
% SeeBeyond - Final End-to-End Pipeline
%
% Stage 1:
%   Image quality / preprocessing
%
% Stage 2:
%   Retinal lesion segmentation
%
% Stage 3:
%   Diabetic Retinopathy severity classification
%
% Input:
%   results = runSeeBeyondPipeline(I)
%
% OR:
%   results = runSeeBeyondPipeline("image.jpg")

%% =========================================================
% PROJECT ROOT
%% =========================================================

thisFile = mfilename('fullpath');

srcDir = fileparts(thisFile);
projectRoot = fileparts(srcDir);

addpath(genpath(fullfile(projectRoot,'src')));

fprintf('\n');
fprintf('====================================================\n');
fprintf('              SEEBEYOND AI SYSTEM\n');
fprintf('====================================================\n');


%% =========================================================
% READ INPUT IMAGE
%% =========================================================

if ischar(inputImage) || isstring(inputImage)

    fprintf('\nInput: %s\n',string(inputImage));

    I = imread(inputImage);

else

    I = inputImage;

end


if size(I,3) == 1

    I = repmat(I,[1 1 3]);

end


fprintf('Original image size : %d x %d\n', ...
    size(I,1),size(I,2));


%% =========================================================
% STAGE 1
%
% Temporary integration:
% Input validation and basic normalization.
%
% We will connect the complete Stage-1 quality pipeline
% after confirming its exact function interface.
%% =========================================================

fprintf('\n====================================================\n');
fprintf('STAGE 1 - IMAGE INPUT\n');
fprintf('====================================================\n');

if ~isfloat(I)

    Iprocessed = im2uint8(I);

else

    Iprocessed = im2uint8(mat2gray(I));

end

fprintf('Input image accepted.\n');


%% =========================================================
% STAGE 2
% RETINAL LESION SEGMENTATION
%% =========================================================

fprintf('\n====================================================\n');
fprintf('STAGE 2 - RETINAL SEGMENTATION\n');
fprintf('====================================================\n');

segResults = runSegmentationPipeline(Iprocessed);


%% =========================================================
% STAGE 3
% DR SEVERITY CLASSIFICATION
%% =========================================================

fprintf('\n====================================================\n');
fprintf('STAGE 3 - DR SEVERITY CLASSIFICATION\n');
fprintf('====================================================\n');


persistent drClassifier
persistent classNames
persistent inputSize


if isempty(drClassifier)

    fprintf('\nLoading Stage-3 classifier...\n');

    S = load( ...
        fullfile(projectRoot,'models','drClassifier.mat'), ...
        'drClassifier', ...
        'classNames', ...
        'inputSize');

    drClassifier = S.drClassifier;
    classNames = S.classNames;
    inputSize = S.inputSize;

    fprintf('DR classifier loaded.\n');

end


%% =========================================================
% PREPARE CLASSIFIER INPUT
%% =========================================================

classifierImage = imresize( ...
    Iprocessed, ...
    inputSize(1:2));


if size(classifierImage,3) == 1

    classifierImage = ...
        repmat(classifierImage,[1 1 3]);

end


%% =========================================================
% CLASSIFIER PREDICTION
%% =========================================================

fprintf('\nRunning DR severity prediction...\n');

scores = minibatchpredict( ...
    drClassifier, ...
    classifierImage, ...
    ExecutionEnvironment="gpu");


if isa(scores,'dlarray')

    scores = extractdata(scores);

end


if isa(scores,'gpuArray')

    scores = gather(scores);

end


scores = squeeze(scores);


%% =========================================================
% CONVERT SCORES TO CLASS
%% =========================================================

[confidence,predictedIndex] = max(scores);

predictedClass = classNames(predictedIndex);


fprintf('\nPredicted DR class : %s\n', ...
    string(predictedClass));

fprintf('Confidence         : %.2f %%\n', ...
    confidence*100);


%% =========================================================
% DR CLASS DESCRIPTION
%% =========================================================

switch string(predictedClass)

    case "No_DR"

        severity = "No Diabetic Retinopathy";

    case "Mild"

        severity = "Mild Diabetic Retinopathy";

    case "Moderate"

        severity = "Moderate Diabetic Retinopathy";

    case "Severe"

        severity = "Severe Diabetic Retinopathy";

    case "Proliferative_DR"

        severity = "Proliferative Diabetic Retinopathy";

    otherwise

        severity = string(predictedClass);

end


%% =========================================================
% CLASS PROBABILITIES
%% =========================================================

classProbability = table( ...
    classNames(:), ...
    scores(:), ...
    'VariableNames', ...
    {'Class','Probability'});


%% =========================================================
% LESION SUMMARY
%% =========================================================

segSummary = segResults.summary;


%% =========================================================
% FINAL RESULT STRUCTURE
%% =========================================================

results = struct;


% Original image
results.inputImage = Iprocessed;


% Stage 2 results
results.segmentation = segResults;


% Classification
results.predictedClass = predictedClass;

results.severity = severity;

results.confidence = confidence;

results.classProbability = classProbability;


% Convenient direct access
results.retinaMask = ...
    segResults.retinaMask;

results.vesselMask = ...
    segResults.vesselMask;

results.hardExudateMask = ...
    segResults.hardExudateMask;

results.haemorrhageMask = ...
    segResults.haemorrhageMask;

results.microaneurysmMask = ...
    segResults.microaneurysmMask;

results.softExudateMask = ...
    segResults.softExudateMask;

results.overlay = ...
    segResults.overlay;

results.summary = segSummary;


%% =========================================================
% FINAL CONSOLE REPORT
%% =========================================================

fprintf('\n');
fprintf('====================================================\n');
fprintf('              SEEBEYOND FINAL RESULT\n');
fprintf('====================================================\n');

fprintf('\nDR Severity : %s\n',severity);

fprintf('Confidence  : %.2f %%\n', ...
    confidence*100);


fprintf('\n----- CLASS PROBABILITIES -----\n');

for k = 1:height(classProbability)

    fprintf('%-20s : %6.2f %%\n', ...
        string(classProbability.Class(k)), ...
        classProbability.Probability(k)*100);

end


fprintf('\n----- DETECTED RETINAL STRUCTURES -----\n');

disp(segSummary);


fprintf('====================================================\n');
fprintf('           SEEBEYOND PIPELINE COMPLETE\n');
fprintf('====================================================\n');

end