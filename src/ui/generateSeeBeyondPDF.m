function generateSeeBeyondPDF(result, imagePath, pdfPath)
% generateSeeBeyondPDF
% Creates a SeeBeyond AI retinal screening PDF report.

import mlreportgen.dom.*

% ---------------------------------------------------------
% Prepare output directory
% ---------------------------------------------------------

[outDir,~,~] = fileparts(pdfPath);

if ~isempty(outDir) && ~exist(outDir,'dir')
    mkdir(outDir);
end

% ---------------------------------------------------------
% Temporary directory
% ---------------------------------------------------------

tmpDir = fullfile(tempdir, ...
    ['SeeBeyond_' datestr(now,'yyyymmdd_HHMMSSFFF')]);

mkdir(tmpDir);

% ---------------------------------------------------------
% Temporary image files
% ---------------------------------------------------------

originalImg = fullfile(tmpDir,'original.png');
analysisImg = fullfile(tmpDir,'analysis.png');
probImg     = fullfile(tmpDir,'probabilities.png');

% ---------------------------------------------------------
% Save original image
% ---------------------------------------------------------

try
    if isfield(result,'image') && ~isempty(result.image)
        imwrite(result.image,originalImg);
    else
        copyfile(imagePath,originalImg);
    end
catch
    copyfile(imagePath,originalImg);
end

% ---------------------------------------------------------
% Save AI analysis image
% ---------------------------------------------------------

try
    if isfield(result,'analysisImage') && ~isempty(result.analysisImage)

        imwrite(result.analysisImage,analysisImg);

    elseif isfield(result,'segmentation') && ...
            isfield(result.segmentation,'overlay') && ...
            ~isempty(result.segmentation.overlay)

        imwrite(result.segmentation.overlay,analysisImg);

    else

        copyfile(imagePath,analysisImg);

    end

catch
    copyfile(imagePath,analysisImg);
end

% ---------------------------------------------------------
% Create probability chart
% ---------------------------------------------------------

fig = figure( ...
    'Visible','off', ...
    'Color','white', ...
    'Position',[100 100 900 500]);

try

    scores = double(result.classScores(:)) * 100;
    names  = cellstr(string(result.classNames));

    barh(scores);

    ax = gca;
    ax.YTick = 1:numel(names);
    ax.YTickLabel = names;
    ax.XLim = [0 100];
    ax.FontSize = 11;
    ax.Box = 'off';

    xlabel('Probability (%)');
    title('AI Diabetic Retinopathy Classification');

    grid on;

    exportgraphics(fig,probImg, ...
        'Resolution',150);

catch ME

    close(fig);

    if exist(tmpDir,'dir')
        try
            rmdir(tmpDir,'s');
        catch
        end
    end

    rethrow(ME);

end

close(fig);

% ---------------------------------------------------------
% Create PDF document
% ---------------------------------------------------------

doc = Document(pdfPath,'pdf');

try

    % =====================================================
    % TITLE
    % =====================================================

    p = mlreportgen.dom.Paragraph('SEEBEYOND');
    p.Bold = true;
    p.FontSize = '24pt';
    p.HAlign = 'center';
    append(doc,p);

    p = mlreportgen.dom.Paragraph('AI-Powered Retinal Screening Report');
    p.FontSize = '14pt';
    p.HAlign = 'center';
    append(doc,p);

    p = mlreportgen.dom.Paragraph('Retinal Image Analysis & Diabetic Retinopathy Screening');
    p.HAlign = 'center';
    append(doc,p);

    append(doc,mlreportgen.dom.Paragraph(' '));

    % =====================================================
    % SCREENING INFORMATION
    % =====================================================

    append(doc,heading('1. SCREENING INFORMATION'));

    reportID = ['SB-' datestr(now,'yyyymmdd-HHMMSS')];

    info = {
        'Report ID',reportID;
        'Analysis Date',datestr(now,'dd/mm/yyyy HH:MM');
        'Image',imagePath;
        'Screening Type','AI Retinal Screening';
        'Status','Completed';
        'AI Pipeline','SeeBeyond AI Pipeline'
        };

    append(doc,mlreportgen.dom.Table(info));

    append(doc,mlreportgen.dom.Paragraph(' '));

    % =====================================================
    % FUNDUS IMAGES
    % =====================================================

    append(doc,heading('2. FUNDUS IMAGE'));

    p = mlreportgen.dom.Paragraph('Original Fundus Image');
    p.Bold = true;
    append(doc,p);

    img = mlreportgen.dom.Image(originalImg);
    img.Width = '2.7in';
    img.Height = '2.0in';
    append(doc,img);

    append(doc,mlreportgen.dom.Paragraph(' '));

    p = mlreportgen.dom.Paragraph('AI Analysis / Lesion Localization');
    p.Bold = true;
    append(doc,p);

    img = mlreportgen.dom.Image(analysisImg);
    img.Width = '2.7in';
    img.Height = '2.0in';
    append(doc,img);

    % =====================================================
    % IMAGE QUALITY
    % =====================================================

    append(doc,heading('3. IMAGE QUALITY ASSESSMENT'));

    qualityStatus = getValue(result,'qualityStatus','Not available');

    blur = 'Not available';
    brightness = 'Not available';
    fov = 'Not available';

    if isfield(result,'qualityReport')

        qr = result.qualityReport;

        if isfield(qr,'blurScore')
            blur = sprintf('%.6f',double(qr.blurScore));
        end

        if isfield(qr,'brightnessScore')
            brightness = sprintf('%.4f',double(qr.brightnessScore));
        end

        if isfield(qr,'fovScore')
            fov = sprintf('%.4f',double(qr.fovScore));
        end

    end

    quality = {
        'Parameter','Value';
        'Overall Quality',qualityStatus;
        'Blur Score',blur;
        'Brightness',brightness;
        'Field of View',fov
        };

    append(doc,mlreportgen.dom.Table(quality));

    % =====================================================
    % AI RESULT
    % =====================================================

    append(doc,heading('4. AI SCREENING RESULT'));

    severity = getValue(result,'severity','Not available');

    grade = 'Not available';

    if isfield(result,'drGrade')
        grade = sprintf('%d / 4',double(result.drGrade));
    end

    confidence = 'Not available';

    if isfield(result,'confidencePercent')
        confidence = sprintf('%.2f%%', ...
            double(result.confidencePercent));
    elseif isfield(result,'confidence')
        confidence = sprintf('%.2f%%', ...
            double(result.confidence)*100);
    end

    resultTable = {
        'Predicted DR Severity',severity;
        'DR Grade',grade;
        'AI Confidence',confidence
        };

    append(doc,mlreportgen.dom.Table(resultTable));

    % =====================================================
    % CLASSIFICATION
    % =====================================================

    append(doc,heading('5. AI CLASSIFICATION BREAKDOWN'));

    img = mlreportgen.dom.Image(probImg);
    img.Width = '2.8in';
    img.Height = '1.9in';
    append(doc,img);

    % Classification graph explanation
    predClass = getValue(result,'drClass','Not available');
    predConfidence = confidence;

    explanation = mlreportgen.dom.Paragraph( ...
        sprintf(['The graph shows the AI probability distribution across the five diabetic retinopathy classes. ' ...
                 'The class with the highest probability is the model prediction. ' ...
                 'For this screening, the predicted class is %s with an AI confidence of %s. ' ...
                 'The smaller percentages represent residual model uncertainty and do not mean that multiple DR grades are diagnosed simultaneously.'], ...
                 char(predClass), char(predConfidence)));

    explanation.FontSize = '9pt';
    explanation.Color = '#444444';
    append(doc,explanation);

    % =====================================================
    % RETINAL FEATURES
    % =====================================================

    append(doc,heading('6. RETINAL FEATURES DETECTED BY AI'));

    featureTable = {
        'Retinal Feature','Pixels','Retinal Area';
        'Blood Vessel','--','--';
        'Hard Exudate','--','--';
        'Haemorrhage','--','--';
        'Microaneurysm','--','--';
        'Soft Exudate','--','--'
        };

    if isfield(result,'segmentation')

        seg = result.segmentation;

        if isfield(seg,'retinaMask')
            totalPixels = nnz(seg.retinaMask);
        else
            totalPixels = numel(seg.vesselMask);
        end

        if totalPixels == 0
            totalPixels = 1;
        end

        features = {
            'Blood Vessel','vesselMask';
            'Hard Exudate','hardExudateMask';
            'Haemorrhage','haemorrhageMask';
            'Microaneurysm','microaneurysmMask';
            'Soft Exudate','softExudateMask'
            };

        for k = 1:size(features,1)

            featureName = features{k,1};
            fieldName = features{k,2};

            if isfield(seg,fieldName)

                px = nnz(seg.(fieldName));
                pct = 100 * px / totalPixels;

                featureTable{k+1,1} = featureName;
                featureTable{k+1,2} = sprintf('%d',px);
                featureTable{k+1,3} = sprintf('%.3f%%',pct);

            end

        end

    end

    append(doc,mlreportgen.dom.Table(featureTable));

    % =====================================================
    % INTERPRETATION
    % =====================================================

    append(doc,heading('7. AI INTERPRETATION'));

    interpretation = sprintf( ...
        ['SeeBeyond AI classified the submitted retinal image as ' ...
         '%s with an estimated AI confidence of %s. ' ...
         'The image-quality assessment was %s. ' ...
         'The AI pipeline also analyzed retinal structures and ' ...
         'lesion-like features to provide an explainable screening result.'], ...
         severity,confidence,qualityStatus);

    append(doc,mlreportgen.dom.Paragraph(interpretation));

    % =====================================================
    % RECOMMENDATION
    % =====================================================

    append(doc,heading('8. RECOMMENDED NEXT STEP'));

    recommendation = [ ...
        'Clinical Review Recommended. ' ...
        'This report is intended to support retinal screening ' ...
        'and early identification of possible abnormalities. ' ...
        'A qualified ophthalmologist or eye-care professional ' ...
        'should review the retinal image and determine the ' ...
        'appropriate clinical diagnosis and follow-up.' ...
        ];

    append(doc,mlreportgen.dom.Paragraph(recommendation));

    % =====================================================
    % DISCLAIMER
    % =====================================================

    append(doc,heading('9. MEDICAL DISCLAIMER'));

    disclaimer = [ ...
        'SeeBeyond is an AI-assisted retinal screening system ' ...
        'and is not a substitute for professional medical diagnosis. ' ...
        'AI predictions may contain errors or false positives or ' ...
        'false negatives. This report should be interpreted by a ' ...
        'qualified healthcare professional together with appropriate ' ...
        'clinical examination.' ...
        ];

    append(doc,mlreportgen.dom.Paragraph(disclaimer));

    append(doc,mlreportgen.dom.Paragraph(' '));

    % =====================================================
    % FOOTER
    % =====================================================

    p = mlreportgen.dom.Paragraph('SEEBEYOND');
    p.Bold = true;
    p.HAlign = 'center';
    append(doc,p);

    p = mlreportgen.dom.Paragraph( ...
        'AI-assisted screening • Not a medical diagnosis');

    p.HAlign = 'center';
    append(doc,p);

    % -----------------------------------------------------
    % CLOSE DOCUMENT
    % -----------------------------------------------------

    close(doc);

catch ME

    try
        close(doc);
    catch
    end

    if exist(pdfPath,'file')
        info = dir(pdfPath);

        if info.bytes == 0
            delete(pdfPath);
        end
    end

    if exist(tmpDir,'dir')
        try
            rmdir(tmpDir,'s');
        catch
        end
    end

    rethrow(ME);

end

% ---------------------------------------------------------
% Cleanup
% ---------------------------------------------------------

if exist(tmpDir,'dir')

    try
        rmdir(tmpDir,'s');
    catch
    end

end

end


function p = heading(text)

p = mlreportgen.dom.Paragraph(text);
p.Bold = true;
p.FontSize = '14pt';

end


function value = getValue(s,field,defaultValue)

if isfield(s,field) && ~isempty(s.(field))
    value = char(string(s.(field)));
else
    value = defaultValue;
end

end
