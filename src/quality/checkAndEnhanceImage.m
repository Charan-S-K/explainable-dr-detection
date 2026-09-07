function [outputImage, qualityStatus, report] = checkAndEnhanceImage(I)
% checkAndEnhanceImage
% Complete Stage-1 fundus image quality assessment.
%
% Checks:
%   1. Blur
%   2. Illumination
%   3. Field of View
%
% Outputs:
%   outputImage   - original or enhanced image
%   qualityStatus - "Acceptable", "Borderline", or "Reject"
%   report        - detailed quality measurements

    %% ---------------------------------------------------------
    % 1. Blur assessment
    %% ---------------------------------------------------------

    blurScore = measureBlur(I);

    % Provisional thresholds.
    % These will be calibrated later using many dataset images.
    severeBlurThreshold = 0.000020;
    borderlineBlurThreshold = 0.000050;

    if blurScore < severeBlurThreshold
        blurStatus = "Too Blurry";

    elseif blurScore < borderlineBlurThreshold
        blurStatus = "Borderline";

    else
        blurStatus = "Acceptable";
    end


    %% ---------------------------------------------------------
    % 2. Illumination assessment
    %% ---------------------------------------------------------

    [brightnessScore, illuminationStatus] = measureIllumination(I);


    %% ---------------------------------------------------------
    % 3. Field-of-view assessment
    %% ---------------------------------------------------------

    [fovScore, fovStatus, fovDetails] = evaluateFOV(I);


    %% ---------------------------------------------------------
    % 4. Final quality decision
    %% ---------------------------------------------------------

    rejectionReasons = strings(0);

    % Severe blur cannot really be repaired by enhancement
    if blurStatus == "Too Blurry"
        rejectionReasons(end+1) = "Image too blurry - recapture required";
    end

    % Poor FOV means important retinal information may be missing
    if fovStatus == "Poor"
        rejectionReasons(end+1) = ...
            "Insufficient retinal field of view - recapture required";
    end

    if illuminationStatus == "Invalid Image"
        rejectionReasons(end+1) = ...
            "Unable to detect valid retinal region";
    end


    %% ---------------------------------------------------------
    % Reject image
    %% ---------------------------------------------------------

    if ~isempty(rejectionReasons)

        qualityStatus = "Reject";

        outputImage = I;

        action = "Recapture image";

    else

        %% Determine if image needs enhancement

        needsEnhancement = ...
            illuminationStatus == "Too Dark" || ...
            illuminationStatus == "Too Bright";

        isBorderline = ...
            blurStatus == "Borderline" || ...
            fovStatus == "Borderline";

        if needsEnhancement

            outputImage = enhanceFundusImage(I);

            qualityStatus = "Borderline";

            action = "Enhanced before AI analysis";

        elseif isBorderline

            outputImage = I;

            qualityStatus = "Borderline";

            action = "Proceed with caution";

        else

            outputImage = I;

            qualityStatus = "Acceptable";

            action = "Proceed to AI screening";

        end
    end


    %% ---------------------------------------------------------
    % Create report structure
    %% ---------------------------------------------------------

    report.blurScore = blurScore;
    report.blurStatus = blurStatus;

    report.brightnessScore = brightnessScore;
    report.illuminationStatus = illuminationStatus;

    report.fovScore = fovScore;
    report.fovStatus = fovStatus;
    report.retinaCoverage = fovDetails.coverage;
    report.centerOffset = fovDetails.centerOffset;

    report.rejectionReasons = rejectionReasons;
    report.action = action;

end