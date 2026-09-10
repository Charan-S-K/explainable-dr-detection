function result = runSeeBeyond(imagePath)
% runSeeBeyond
% SEEBEYOND end-to-end retinal analysis pipeline.
%
% Pipeline:
%   1. Image quality assessment and enhancement
%   2. Diabetic retinopathy classification
%   3. Retinal lesion segmentation
%   4. Final structured AI report
%
% IMPORTANT:
%   DR classification is the primary severity result.
%   Segmentation outputs are AI localization/explainability outputs.
%   They are NOT independent clinical diagnoses.

%% =========================================================
%% PROJECT SETUP
%% =========================================================

projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(fullfile(projectRoot,'src')));
    
fprintf('\n');
fprintf('============================================================\n');
fprintf('                    SEEBEYOND AI REPORT\n');
fprintf('============================================================\n');

%% =========================================================
%% INPUT IMAGE
%% =========================================================

if nargin < 1
    error('Please provide an image path.');
end

if ~isfile(imagePath)
    error('Image file not found:\n%s',imagePath);
end

I = imread(imagePath);

if size(I,3) == 1
    I = repmat(I,[1 1 3]);
end

if ~isa(I,'uint8')
    I = im2uint8(I);
end

fprintf('\nInput image:\n%s\n',imagePath);
fprintf('Image size: %d x %d\n',size(I,1),size(I,2));

%% =========================================================
%% STAGE 1 - IMAGE QUALITY ASSESSMENT
%% =========================================================

fprintf('\n');
fprintf('------------------------------------------------------------\n');
fprintf('1. IMAGE QUALITY ASSESSMENT\n');
fprintf('------------------------------------------------------------\n');

try
    [analysisImage,qualityStatus,qualityReport] = checkAndEnhanceImage(I);
    qualityAvailable = true;
catch ME
    warning('Stage-1 quality assessment failed: %s',ME.message);
    analysisImage = I;
    qualityStatus = 'Unavailable';
    qualityReport = struct();
    qualityAvailable = false;
end

fprintf('\nQuality status : %s\n',qualityStatus);

if qualityAvailable
    fprintf('Blur score     : %.6f\n',qualityReport.blurScore);
    fprintf('Blur status    : %s\n',qualityReport.blurStatus);
    fprintf('Brightness     : %.4f\n',qualityReport.brightnessScore);
    fprintf('Illumination   : %s\n',qualityReport.illuminationStatus);
    fprintf('FOV score      : %.4f\n',qualityReport.fovScore);
    fprintf('FOV status     : %s\n',qualityReport.fovStatus);
    fprintf('Action         : %s\n',qualityReport.action);
end

analysisWasEnhanced = ~isequal(analysisImage,I);

if analysisWasEnhanced
    fprintf('\nStage-1 enhancement applied before AI analysis.\n');
else
    fprintf('\nNo Stage-1 enhancement required.\n');
end

%% =========================================================
%% LOAD DR CLASSIFIER
%% =========================================================

fprintf('\n');
fprintf('------------------------------------------------------------\n');
fprintf('LOADING DR CLASSIFIER\n');
fprintf('------------------------------------------------------------\n');

classifierFile = fullfile(projectRoot,'models','drClassifier.mat');

if ~isfile(classifierFile)
    error('DR classifier not found:\n%s',classifierFile);
end

S = load(classifierFile,'drClassifier','classNames','inputSize');
drClassifier = S.drClassifier;
classNames = S.classNames;
inputSize = S.inputSize;

fprintf('DR classifier loaded successfully.\n');

%% =========================================================
%% STAGE 2 - DIABETIC RETINOPATHY CLASSIFICATION
%% =========================================================

fprintf('\n');
fprintf('------------------------------------------------------------\n');
fprintf('2. DIABETIC RETINOPATHY CLASSIFICATION\n');
fprintf('------------------------------------------------------------\n');

% IMPORTANT:
% The DR classifier must use the SAME preprocessing pipeline
% that was used during Stage-3 validation.
%
% Stage-1 enhancement is used for segmentation/explainability,
% but DR classification receives the ORIGINAL fundus image.

try

    classificationResult = predictDRSeverity( ...
        I, ...
        drClassifier);

    drClass = classificationResult.class;
    confidence = classificationResult.confidence;
    scores = classificationResult.probabilities;
    classNames = classificationResult.classNames;

catch ME

    error('DR classification failed: %s',ME.message);

end

fprintf('\nPredicted severity : %s\n',char(drClass));
fprintf('Confidence         : %.2f %%\n',confidence*100);

fprintf('\nClass probabilities:\n');

for k = 1:numel(classNames)

    fprintf('  %-20s %.2f %%\n', ...
        char(classNames(k)), ...
        scores(k)*100);

end

%% =========================================================
%% DR GRADE
%% =========================================================

switch char(drClass)
    case 'No_DR'
        drGrade = 0;
        severityText = 'No Diabetic Retinopathy';
    case 'Mild'
        drGrade = 1;
        severityText = 'Mild Diabetic Retinopathy';
    case 'Moderate'
        drGrade = 2;
        severityText = 'Moderate Diabetic Retinopathy';
    case 'Severe'
        drGrade = 3;
        severityText = 'Severe Diabetic Retinopathy';
    case 'Proliferative_DR'
        drGrade = 4;
        severityText = 'Proliferative Diabetic Retinopathy';
    otherwise
        drGrade = index - 1;
        severityText = char(drClass);
end

fprintf('\nPredicted severity : %s\n',severityText);
fprintf('DR grade           : %d\n',drGrade);
fprintf('Confidence         : %.2f %%\n',confidence*100);

fprintf('\nClass probabilities:\n');

for k = 1:numel(classNames)
    fprintf('  %-20s %.2f %%\n',char(classNames(k)),scores(k)*100);
end

%% =========================================================
%% STAGE 3 - RETINAL LESION SEGMENTATION
%% =========================================================

fprintf('\n');
fprintf('------------------------------------------------------------\n');
fprintf('3. RETINAL LESION SEGMENTATION\n');
fprintf('------------------------------------------------------------\n');

try
    segmentation = runSegmentationPipeline(analysisImage);
    segmentationAvailable = true;
catch ME
    warning('Stage-2 segmentation failed: %s',ME.message);
    segmentation = struct();
    segmentationAvailable = false;
end

%% =========================================================
%% EXTRACT SEGMENTATION STATISTICS
%% =========================================================

lesionNames = {
    'Blood Vessel'
    'Hard Exudate'
    'Haemorrhage'
    'Microaneurysm'
    'Soft Exudate'
    };

pixelCounts = zeros(5,1);
retinaPercent = zeros(5,1);

if segmentationAvailable && isfield(segmentation,'summary')
    summaryTable = segmentation.summary;
    try
        pixelCounts = summaryTable.PixelCount;
        retinaPercent = summaryTable.RetinaPercent;
    catch
        % Keep zeros if summary format is different.
    end
end

%% =========================================================
%% AI LOCALIZATION STATUS
%% =========================================================

aiLocalized = false(5,1);

if segmentationAvailable

    if isfield(segmentation,'vesselMask')
        aiLocalized(1) = any(segmentation.vesselMask(:));
    end

    if isfield(segmentation,'hardExudateMask')
        aiLocalized(2) = any(segmentation.hardExudateMask(:));
    end

    if isfield(segmentation,'haemorrhageMask')
        aiLocalized(3) = any(segmentation.haemorrhageMask(:));
    end

    if isfield(segmentation,'microaneurysmMask')
        aiLocalized(4) = any(segmentation.microaneurysmMask(:));
    end

    if isfield(segmentation,'softExudateMask')
        aiLocalized(5) = any(segmentation.softExudateMask(:));
    end

end

%% =========================================================
%% FINAL AI ASSESSMENT
%% =========================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                    FINAL AI ASSESSMENT\n');
fprintf('============================================================\n');

fprintf('\nImage Quality\n');
fprintf('-------------\n');
fprintf('Status     : %s\n',qualityStatus);

fprintf('\nDR Assessment\n');
fprintf('-------------\n');
fprintf('Severity   : %s\n',severityText);
fprintf('DR Grade   : %d / 4\n',drGrade);
fprintf('Confidence : %.2f %%\n',confidence*100);

fprintf('\nAI Lesion Localization\n');
fprintf('----------------------\n');

for k = 1:5
    if aiLocalized(k)
        status = 'AI-localized';
    else
        status = 'No pixels localized';
    end

    fprintf('%-18s : %-18s (%d pixels, %.3f%% retina)\n', ...
        lesionNames{k},status,pixelCounts(k),retinaPercent(k));
end

fprintf('\nNote: Segmentation outputs represent AI localization for\n');
fprintf('explainability and are not independent clinical diagnoses.\n');

%% =========================================================
%% STRUCTURED SUMMARY
%% =========================================================

summary = struct();
summary.qualityStatus = qualityStatus;
summary.qualityReport = qualityReport;
summary.analysisWasEnhanced = analysisWasEnhanced;
summary.severity = severityText;
summary.drGrade = drGrade;
summary.confidence = confidence;
summary.confidencePercent = confidence*100;
summary.lesionNames = lesionNames;
summary.pixelCounts = pixelCounts;
summary.retinaPercent = retinaPercent;
summary.aiLocalized = aiLocalized;
summary.screeningType = 'AI-assisted retinal screening prototype';
summary.disclaimer = 'Segmentation outputs are AI localization/explainability signals and not independent clinical diagnoses.';

%% =========================================================
%% BUILD OUTPUT STRUCTURE
%% =========================================================

result = struct();

result.image = I;
result.imagePath = imagePath;

result.analysisImage = analysisImage;
result.qualityStatus = qualityStatus;
result.qualityReport = qualityReport;
result.analysisWasEnhanced = analysisWasEnhanced;

result.drClass = drClass;
result.drGrade = drGrade;
result.severity = severityText;
result.confidence = confidence;
result.confidencePercent = confidence*100;
result.classScores = scores;
result.classNames = classNames;

result.segmentation = segmentation;
result.segmentationAvailable = segmentationAvailable;
result.aiLocalized = aiLocalized;
result.summary = summary;

%% =========================================================
%% VISUALIZATION
%% =========================================================

figure('Name','SEEBEYOND AI REPORT','NumberTitle','off','Color','w','Position',[100 100 1400 700]);

subplot(2,3,1);
imshow(I);
title('Original Fundus Image');

subplot(2,3,2);
imshow(analysisImage);

if analysisWasEnhanced
    title('Stage-1 Enhanced Image');
else
    title('Stage-1 Analysis Image');
end

subplot(2,3,3);

if segmentationAvailable && isfield(segmentation,'overlay')
    imshow(segmentation.overlay);
    title('AI Lesion Localization');
else
    imshow(analysisImage);
    title('Segmentation unavailable');
end

subplot(2,3,4);
bar(scores*100);
xticks(1:numel(classNames));
xticklabels({'No DR','Mild','Moderate','Severe','Proliferative'});
xtickangle(30);
ylabel('Probability (%)');
title('DR Classification Probability');
ylim([0 100]);
grid on;

subplot(2,3,5);
bar(pixelCounts);
xticks(1:5);
xticklabels({'Vessel','Hard EX','HE','MA','Soft EX'});
xtickangle(30);
ylabel('Pixels');
title('AI-Localized Pixel Count');
grid on;

subplot(2,3,6);
axis off;

text(0.05,0.90,'SEEBEYOND RESULT','FontSize',18,'FontWeight','bold');
text(0.05,0.72,sprintf('Severity:\n%s',severityText),'FontSize',14);
text(0.05,0.48,sprintf('Confidence: %.2f%%',confidence*100),'FontSize',14);
text(0.05,0.30,sprintf('DR Grade: %d / 4',drGrade),'FontSize',14);
text(0.05,0.12,sprintf('Quality: %s',qualityStatus),'FontSize',11);

sgtitle('SEEBEYOND - AI RETINAL SCREENING SYSTEM','FontSize',18,'FontWeight','bold');

%% =========================================================
%% CONSOLE COMPLETION
%% =========================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('             SEEBEYOND ANALYSIS COMPLETE\n');
fprintf('============================================================\n');

fprintf('\nFinal Result:\n');
fprintf('  Image Quality : %s\n',qualityStatus);
fprintf('  DR Severity   : %s\n',severityText);
fprintf('  DR Grade      : %d / 4\n',drGrade);
fprintf('  Confidence    : %.2f %%\n',confidence*100);

end