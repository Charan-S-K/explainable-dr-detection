function risk = assessScreeningRisk(result)
% assessScreeningRisk
% SeeBeyond AI-assisted retinal screening risk assessment.
%
% Prototype screening-prioritization layer.
% NOT a clinically validated medical risk prediction model.

risk = struct();

risk.score = 0;
risk.level = "LOW";
risk.title = "Low Screening Risk";
risk.reason = "No major risk indicators identified by the AI screening.";
risk.recommendation = "Continue routine eye screening and monitoring.";
risk.factors = strings(0,1);

risk.drGrade = 0;
risk.confidence = 0;

risk.disclaimer = ...
    "This risk assessment is AI-assisted screening guidance and is not a medical diagnosis. Clinical decisions should be made by a qualified eye-care professional.";

% ============================================================
% VALIDATE INPUT
% ============================================================

if nargin < 1 || isempty(result) || ~isstruct(result)

    risk.level = "UNKNOWN";
    risk.title = "Risk Assessment Unavailable";
    risk.reason = "Valid screening results were not available.";
    risk.recommendation = "Repeat the screening with a valid retinal image.";

    return;
end

% ============================================================
% DR GRADE
% ============================================================

if isfield(result,"drGrade") && ~isempty(result.drGrade)
    drGrade = double(result.drGrade);
else
    drGrade = 0;
end

drGrade = round(drGrade);
drGrade = max(0,min(4,drGrade));

risk.drGrade = drGrade;

% DR contribution
gradeScores = [0 20 40 65 85];

risk.score = risk.score + gradeScores(drGrade + 1);

if drGrade > 0

    risk.factors(end+1) = sprintf( ...
        "Diabetic retinopathy grade %d/4 identified", ...
        drGrade);

end

% ============================================================
% AI CONFIDENCE
% ============================================================

if isfield(result,"confidence") && ~isempty(result.confidence)

    confidence = double(result.confidence);

elseif isfield(result,"confidencePercent") && ...
        ~isempty(result.confidencePercent)

    confidence = double(result.confidencePercent) / 100;

else

    confidence = 0;

end

if confidence > 1
    confidence = confidence / 100;
end

confidence = max(0,min(1,confidence));

risk.confidence = confidence;

% Low confidence adds uncertainty points.
if confidence < 0.50

    risk.score = risk.score + 10;

    risk.factors(end+1) = ...
        "Lower AI classification confidence";

end

% ============================================================
% RETINAL LESION FINDINGS
% ============================================================

lesionCount = 0;

if isfield(result,"segmentation") && ...
        ~isempty(result.segmentation)

    seg = result.segmentation;

    % IMPORTANT:
    % These are the actual fields returned by SeeBeyond.
    lesionFields = { ...
        "hardExudateMask", ...
        "haemorrhageMask", ...
        "microaneurysmMask", ...
        "softExudateMask"};

    lesionLabels = { ...
        "Hard exudate", ...
        "Haemorrhage", ...
        "Microaneurysm", ...
        "Soft exudate"};

    for k = 1:numel(lesionFields)

        fieldName = lesionFields{k};

        if ~isfield(seg,fieldName)
            continue;
        end

        lesionData = seg.(fieldName);

        if isempty(lesionData)
            continue;
        end

        % Actual segmentation output is a logical mask.
        pixelCount = nnz(lesionData);

        if pixelCount > 0

            lesionCount = lesionCount + 1;

            risk.factors(end+1) = sprintf( ...
                "%s detected by AI (%d pixels)", ...
                lesionLabels{k}, ...
                pixelCount);

        end

    end

end

% Maximum lesion contribution = 20 points.
lesionScore = min(20,lesionCount * 5);

risk.score = risk.score + lesionScore;

% ============================================================
% IMAGE QUALITY
% ============================================================

if isfield(result,"qualityStatus") && ...
        ~isempty(result.qualityStatus)

    qualityStatus = lower(string(result.qualityStatus));

    if contains(qualityStatus,"poor") || ...
            contains(qualityStatus,"unacceptable")

        risk.score = risk.score + 10;

        risk.factors(end+1) = ...
            "Poor image quality may reduce screening reliability";

    end

end

% ============================================================
% LIMIT SCORE
% ============================================================

risk.score = round(max(0,min(100,risk.score)));

% ============================================================
% RISK LEVEL
% ============================================================

if risk.score >= 70

    risk.level = "HIGH";
    risk.title = "High Screening Risk";

    risk.reason = ...
        "The AI screening identified findings that warrant prompt clinical review.";

    risk.recommendation = ...
        "Prompt ophthalmology evaluation is recommended.";

elseif risk.score >= 35

    risk.level = "MODERATE";
    risk.title = "Moderate Screening Risk";

    risk.reason = ...
        "The AI screening identified findings that warrant clinical review.";

    risk.recommendation = ...
        "Clinical ophthalmology review is recommended.";

else

    risk.level = "LOW";
    risk.title = "Low Screening Risk";

    risk.reason = ...
        "The AI screening did not identify major risk indicators.";

    risk.recommendation = ...
        "Continue routine eye screening and monitoring.";

end

% ============================================================
% FACTORS
% ============================================================

if isempty(risk.factors)

    risk.factors = ...
        "No additional AI risk indicators identified";

end

% ============================================================
% DISPLAY VALUES
% ============================================================

risk.scoreText = sprintf("%d / 100",risk.score);

risk.confidenceText = sprintf( ...
    "%.2f%%", ...
    risk.confidence * 100);

risk.drGradeText = sprintf( ...
    "%d / 4", ...
    risk.drGrade);

end
