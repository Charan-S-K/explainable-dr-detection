function SeeBeyondApp
% SeeBeyondApp
% SeeBeyond - AI-Assisted Retinal Screening
%
% This UI is only the presentation layer.
% The validated AI backend runSeeBeyond() is not modified.

%% ============================================================
% PROJECT SETUP
% =============================================================

projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(fullfile(projectRoot,'src')));
rehash;

selectedImagePath = "";
lastResult = [];

%% ============================================================
% THEME
% =============================================================

BG        = [0.95 0.97 0.98];
WHITE     = [1.00 1.00 1.00];
NAVY      = [0.07 0.12 0.19];
TEXT      = [0.10 0.14 0.20];
SECONDARY = [0.38 0.43 0.50];
BLUE      = [0.10 0.42 0.75];
GREEN     = [0.10 0.55 0.30];
ORANGE    = [0.88 0.55 0.08];
RED       = [0.75 0.18 0.18];
LIGHTBLUE = [0.91 0.95 0.99];
LIGHTGRAY = [0.91 0.93 0.95];
BORDER    = [0.83 0.87 0.90];

%% ============================================================
% MAIN WINDOW
% =============================================================

fig = uifigure( ...
    'Name','SeeBeyond - AI Retinal Screening', ...
    'Position',[60 40 1500 900], ...
    'Color',BG);

%% ============================================================
% HEADER
% =============================================================

header = uipanel(fig, ...
    'Position',[0 825 1500 75], ...
    'BorderType','none', ...
    'BackgroundColor',NAVY);

uilabel(header, ...
    'Position',[30 31 300 30], ...
    'Text','SEEBEYOND', ...
    'FontSize',25, ...
    'FontWeight','bold', ...
    'FontColor',WHITE);

uilabel(header, ...
    'Position',[32 9 600 22], ...
    'Text','AI-Powered Retinal Screening', ...
    'FontSize',12, ...
    'FontColor',[0.82 0.87 0.92]);

statusLabel = uilabel(header, ...
    'Position',[1040 25 420 30], ...
    'Text','● SYSTEM READY', ...
    'HorizontalAlignment','right', ...
    'FontSize',13, ...
    'FontWeight','bold', ...
    'FontColor',[0.55 0.95 0.65]);

%% ============================================================
% LEFT - RETINAL IMAGE
% =============================================================

imageCard = uipanel(fig, ...
    'Position',[25 455 650 345], ...
    'BackgroundColor',WHITE, ...
    'BorderType','line', ...
    'HighlightColor',BORDER);

uilabel(imageCard, ...
    'Position',[22 307 400 25], ...
    'Text','1. RETINAL IMAGE', ...
    'FontSize',13, ...
    'FontWeight','bold', ...
    'FontColor',TEXT);

uilabel(imageCard, ...
    'Position',[22 282 590 20], ...
    'Text','Upload a retinal fundus photograph to begin AI screening.', ...
    'FontSize',10, ...
    'FontColor',SECONDARY);

imageAxes = uiaxes(imageCard, ...
    'Position',[22 48 606 220], ...
    'Color',WHITE);

imageAxes.XTick = [];
imageAxes.YTick = [];
imageAxes.Box = 'on';
imageAxes.Toolbar.Visible = 'off';

title(imageAxes,'No image selected', ...
    'Color',SECONDARY);

imageInfoLabel = uilabel(imageCard, ...
    'Position',[22 16 606 24], ...
    'Text','No image selected', ...
    'HorizontalAlignment','center', ...
    'FontSize',10, ...
    'FontColor',SECONDARY);

%% ============================================================
% RIGHT TOP - IMAGE QUALITY
% =============================================================

qualityCard = uipanel(fig, ...
    'Position',[700 675 775 125], ...
    'BackgroundColor',WHITE, ...
    'BorderType','line', ...
    'HighlightColor',BORDER);

uilabel(qualityCard, ...
    'Position',[22 82 250 22], ...
    'Text','2. IMAGE QUALITY', ...
    'FontSize',11, ...
    'FontWeight','bold', ...
    'FontColor',SECONDARY);

qualityValue = uilabel(qualityCard, ...
    'Position',[22 42 300 34], ...
    'Text','Waiting for image', ...
    'FontSize',19, ...
    'FontWeight','bold', ...
    'FontColor',TEXT);

qualityDetails = uilabel(qualityCard, ...
    'Position',[350 42 390 30], ...
    'Text','Blur: --   |   Brightness: --   |   FOV: --', ...
    'FontSize',10, ...
    'FontColor',SECONDARY);

%% ============================================================
% RIGHT MIDDLE - PRIMARY AI RESULT
% =============================================================

drCard = uipanel(fig, ...
    'Position',[700 455 775 200], ...
    'BackgroundColor',WHITE, ...
    'BorderType','line', ...
    'HighlightColor',BORDER);

uilabel(drCard, ...
    'Position',[22 160 450 22], ...
    'Text','3. AI SCREENING RESULT', ...
    'FontSize',11, ...
    'FontWeight','bold', ...
    'FontColor',SECONDARY);

uilabel(drCard, ...
    'Position',[22 127 180 20], ...
    'Text','DIABETIC RETINOPATHY', ...
    'FontSize',10, ...
    'FontWeight','bold', ...
    'FontColor',SECONDARY);

severityValue = uilabel(drCard, ...
    'Position',[22 83 400 45], ...
    'Text','Waiting for analysis', ...
    'FontSize',23, ...
    'FontWeight','bold', ...
    'FontColor',TEXT);

gradeValue = uilabel(drCard, ...
    'Position',[22 48 250 25], ...
    'Text','Grade: -- / 4', ...
    'FontSize',12, ...
    'FontColor',SECONDARY);

uilabel(drCard, ...
    'Position',[440 127 200 20], ...
    'Text','AI CONFIDENCE', ...
    'FontSize',10, ...
    'FontWeight','bold', ...
    'FontColor',SECONDARY);

confidenceValue = uilabel(drCard, ...
    'Position',[440 82 250 45], ...
    'Text','--', ...
    'FontSize',27, ...
    'FontWeight','bold', ...
    'FontColor',BLUE);

uilabel(drCard, ...
    'Position',[440 48 300 25], ...
    'Text','Probability of the predicted class', ...
    'FontSize',10, ...
    'FontColor',SECONDARY);

%% ============================================================
% BOTTOM LEFT - CLASSIFICATION
% =============================================================

probCard = uipanel(fig, ...
    'Position',[25 185 650 250], ...
    'BackgroundColor',WHITE, ...
    'BorderType','line', ...
    'HighlightColor',BORDER);

uilabel(probCard, ...
    'Position',[22 215 600 22], ...
    'Text','4. HOW THE AI CLASSIFIED THE IMAGE', ...
    'FontSize',11, ...
    'FontWeight','bold', ...
    'FontColor',TEXT);

uilabel(probCard, ...
    'Position',[22 193 600 18], ...
    'Text','Higher bars indicate classes the model considers more likely.', ...
    'FontSize',9, ...
    'FontColor',SECONDARY);

probAxes = uiaxes(probCard, ...
    'Position',[22 25 606 165], ...
    'Color',WHITE);

probAxes.Box = 'on';
probAxes.Toolbar.Visible = 'off';

%% ============================================================
% BOTTOM RIGHT - LESION LOCALIZATION
% =============================================================

lesionCard = uipanel(fig, ...
    'Position',[700 185 775 250], ...
    'BackgroundColor',WHITE, ...
    'BorderType','line', ...
    'HighlightColor',BORDER);

uilabel(lesionCard, ...
    'Position',[22 215 600 22], ...
    'Text','5. RETINAL FEATURES DETECTED BY AI', ...
    'FontSize',11, ...
    'FontWeight','bold', ...
    'FontColor',TEXT);

uilabel(lesionCard, ...
    'Position',[22 193 720 18], ...
    'Text','Highlighted areas help explain which retinal features were detected.', ...
    'FontSize',9, ...
    'FontColor',SECONDARY);

lesionAxes = uiaxes(lesionCard, ...
    'Position',[22 25 450 160], ...
    'Color',WHITE);

lesionAxes.Box = 'on';
lesionAxes.Toolbar.Visible = 'off';

localizationStatus = uitextarea(lesionCard, ...
    'Position',[490 28 260 158], ...
    'Editable','off', ...
    'FontSize',10, ...
    'FontColor',TEXT, ...
    'BackgroundColor',WHITE, ...
    'Value',{ ...
    'AI feature localization'
    ''
    'No analysis available.'
    ''
    'Run AI analysis to'
    'view detected features.'
    });

%% ============================================================
% CONTROL BAR
% =============================================================

controlPanel = uipanel(fig, ...
    'Position',[25 105 1450 60], ...
    'BackgroundColor',WHITE, ...
    'BorderType','line', ...
    'HighlightColor',BORDER);

selectButton = uibutton(controlPanel, ...
    'push', ...
    'Position',[20 12 200 36], ...
    'Text','UPLOAD FUNDUS IMAGE', ...
    'FontSize',11, ...
    'FontWeight','bold', ...
    'FontColor',WHITE, ...
    'BackgroundColor',BLUE, ...
    'ButtonPushedFcn',@selectImage);

analyzeButton = uibutton(controlPanel, ...
    'push', ...
    'Position',[235 12 175 36], ...
    'Text','RUN AI ANALYSIS', ...
    'FontSize',11, ...
    'FontWeight','bold', ...
    'FontColor',WHITE, ...
    'BackgroundColor',GREEN, ...
    'Enable','off', ...
    'ButtonPushedFcn',@analyzeImage);

clearButton = uibutton(controlPanel, ...
    'push', ...
    'Position',[425 12 110 36], ...
    'Text','CLEAR', ...
    'FontSize',11, ...
    'FontColor',TEXT, ...
    'BackgroundColor',LIGHTGRAY, ...
    'ButtonPushedFcn',@clearResults);

saveButton = uibutton(controlPanel, ...
    'push', ...
    'Position',[550 12 150 36], ...
    'Text','SAVE REPORT', ...
    'FontSize',11, ...
    'FontWeight','bold', ...
    'FontColor',TEXT, ...
    'BackgroundColor',LIGHTGRAY, ...
    'Enable','off', ...
    'ButtonPushedFcn',@saveReport);

uilabel(controlPanel, ...
    'Position',[820 12 600 36], ...
    'Text','AI-assisted screening prototype  |  Not a clinical diagnosis', ...
    'HorizontalAlignment','right', ...
    'FontSize',10, ...
    'FontColor',SECONDARY);

%% ============================================================
% FOOTER
% =============================================================

footer = uipanel(fig, ...
    'Position',[25 20 1450 65], ...
    'BorderType','none', ...
    'BackgroundColor',BG);

uilabel(footer, ...
    'Position',[0 35 1000 22], ...
    'Text','SeeBeyond  •  Image Quality  •  DR Classification  •  AI Feature Localization', ...
    'FontSize',10, ...
    'FontColor',SECONDARY);

uilabel(footer, ...
    'Position',[0 8 1400 22], ...
    'Text','Localization results are AI explainability signals and should not be interpreted as independent clinical diagnoses.', ...
    'FontSize',9, ...
    'FontColor',SECONDARY);

%% ============================================================
% SELECT IMAGE
% ============================================================

function selectImage(~,~)

    [file,path] = uigetfile( ...
        {'*.jpg;*.jpeg;*.png;*.tif;*.tiff','Fundus Images'; ...
         '*.*','All Files'}, ...
        'Select Retinal Fundus Image');

    if isequal(file,0)
        return;
    end

    selectedImagePath = string(fullfile(path,file));

    try

        I = imread(char(selectedImagePath));

        if size(I,3) == 1
            I = repmat(I,[1 1 3]);
        end

        imshow(I,'Parent',imageAxes);

        title(imageAxes, ...
            'Original Fundus Image', ...
            'Color',TEXT);

        imageInfoLabel.Text = sprintf( ...
            '%s   |   %d × %d pixels', ...
            file,size(I,2),size(I,1));

        statusLabel.Text = '● IMAGE READY';
        statusLabel.FontColor = ORANGE;

        analyzeButton.Enable = 'on';
        saveButton.Enable = 'off';

        qualityValue.Text = 'Ready for analysis';
        qualityValue.FontColor = TEXT;

        qualityDetails.Text = ...
            'Blur: --   |   Brightness: --   |   FOV: --';

        severityValue.Text = 'Waiting for analysis';
        severityValue.FontColor = TEXT;

        confidenceValue.Text = '--';
        confidenceValue.FontColor = BLUE;

        gradeValue.Text = 'Grade: -- / 4';

        cla(probAxes);
        title(probAxes,'');

        cla(lesionAxes);
        title(lesionAxes,'');

        localizationStatus.Value = { ...
            'AI feature localization'
            ''
            'Run AI analysis to'
            'view detected features.'
            };

        lastResult = [];

    catch ME

        uialert(fig,ME.message,'Image Loading Error');

    end

end

%% ============================================================
% ANALYZE IMAGE
% ============================================================

function analyzeImage(~,~)

    if strlength(selectedImagePath) == 0

        uialert(fig, ...
            'Please upload a fundus image first.', ...
            'No Image');

        return;

    end

    statusLabel.Text = '● AI ANALYSIS RUNNING';
    statusLabel.FontColor = ORANGE;

    analyzeButton.Enable = 'off';
    selectButton.Enable = 'off';

    drawnow;

    try

        % ====================================================
        % VALIDATED SEEBEYOND AI PIPELINE
        % ====================================================

        result = runSeeBeyond(char(selectedImagePath));

        lastResult = result;

        % ====================================================
        % IMAGE / OVERLAY
        % ====================================================

        if isfield(result,'segmentationAvailable') && ...
                result.segmentationAvailable && ...
                isfield(result.segmentation,'overlay')

            imshow(result.segmentation.overlay, ...
                'Parent',imageAxes);

            title(imageAxes, ...
                'AI Feature Localization Overlay', ...
                'Color',TEXT);

        else

            imshow(result.image, ...
                'Parent',imageAxes);

            title(imageAxes, ...
                'Original Fundus Image', ...
                'Color',TEXT);

        end

        % ====================================================
        % IMAGE QUALITY
        % ====================================================

        qualityValue.Text = char(result.qualityStatus);

        if strcmpi(char(result.qualityStatus),'Acceptable')

            qualityValue.FontColor = GREEN;

        elseif strcmpi(char(result.qualityStatus),'Borderline')

            qualityValue.FontColor = ORANGE;

        else

            qualityValue.FontColor = RED;

        end

        if isfield(result,'qualityReport') && ...
                ~isempty(fieldnames(result.qualityReport))

            qr = result.qualityReport;

            qualityDetails.Text = sprintf( ...
                'Blur: %.5f   |   Brightness: %.3f   |   FOV: %.3f', ...
                qr.blurScore, ...
                qr.brightnessScore, ...
                qr.fovScore);

        end

        % ====================================================
        % DR RESULT
        % ====================================================

        severityValue.Text = char(result.severity);

        confidenceValue.Text = sprintf( ...
            '%.2f%%', ...
            result.confidencePercent);

        gradeValue.Text = sprintf( ...
            'Grade: %d / 4', ...
            result.drGrade);

        % Make severity visually meaningful

        severityText = lower(char(result.severity));

        if contains(severityText,'no dr')

            severityValue.FontColor = GREEN;

        elseif contains(severityText,'mild')

            severityValue.FontColor = BLUE;

        elseif contains(severityText,'moderate')

            severityValue.FontColor = ORANGE;

        else

            severityValue.FontColor = RED;

        end

        % ====================================================
        % DR PROBABILITY CHART
        % ====================================================

        cla(probAxes);

        scores = result.classScores(:) * 100;

        bar(probAxes,scores);

        probAxes.XTick = 1:numel(result.classNames);

        probAxes.XTickLabel = { ...
            'No DR'
            'Mild'
            'Moderate'
            'Severe'
            'Proliferative'};

        probAxes.XTickLabelRotation = 25;

        ylabel(probAxes,'Probability (%)');

        ylim(probAxes,[0 max(100,max(scores)*1.15)]);

        title(probAxes, ...
            'AI classification probabilities');

        grid(probAxes,'on');

        % ====================================================
        % LESION / FEATURE SUMMARY
        % ====================================================

        cla(lesionAxes);

        if isfield(result,'segmentation') && ...
                ~isempty(result.segmentation)

            seg = result.segmentation;

            names = { ...
                'Blood Vessel'
                'Hard Exudate'
                'Haemorrhage'
                'Microaneurysm'
                'Soft Exudate'};

            fields = { ...
                'vesselMask'
                'hardExudateMask'
                'haemorrhageMask'
                'microaneurysmMask'
                'softExudateMask'};

            values = zeros(1,numel(fields));

            totalPixels = numel(seg.retinaMask);

            for k = 1:numel(fields)

                if isfield(seg,fields{k})

                    mask = seg.(fields{k});

                    values(k) = 100 * nnz(mask) / totalPixels;

                end

            end

            barh(lesionAxes,values);

            lesionAxes.YTick = 1:numel(names);
            lesionAxes.YTickLabel = names;

            xlabel(lesionAxes,'Retinal area (%)');

            title(lesionAxes,'Detected retinal features');

            grid(lesionAxes,'on');

            localizationStatus.Value = { ...
                'AI feature localization'
                ''
                sprintf('Blood Vessel      %.2f%%',values(1))
                sprintf('Hard Exudate      %.2f%%',values(2))
                sprintf('Haemorrhage       %.2f%%',values(3))
                sprintf('Microaneurysm     %.2f%%',values(4))
                sprintf('Soft Exudate      %.2f%%',values(5))
                ''
                'These values show areas'
                'highlighted by the AI.'
                };

        else

            localizationStatus.Value = { ...
                'AI feature localization'
                ''
                'Segmentation output'
                'is not available.'
                };

        end

        % ====================================================
        % FINAL STATUS
        % ====================================================

        statusLabel.Text = '● ANALYSIS COMPLETE';
        statusLabel.FontColor = GREEN;

        selectButton.Enable = 'on';
        analyzeButton.Enable = 'on';
        saveButton.Enable = 'on';

    catch ME

        statusLabel.Text = '● ANALYSIS ERROR';
        statusLabel.FontColor = RED;

        selectButton.Enable = 'on';
        analyzeButton.Enable = 'on';

        uialert(fig, ...
            ME.message, ...
            'SeeBeyond Analysis Error');

    end

end

%% ============================================================
% CLEAR RESULTS
% ============================================================

function clearResults(~,~)

    selectedImagePath = "";
    lastResult = [];

    cla(imageAxes);

    title(imageAxes, ...
        'No image selected', ...
        'Color',SECONDARY);

    imageInfoLabel.Text = 'No image selected';

    qualityValue.Text = 'Waiting for image';
    qualityValue.FontColor = TEXT;

    qualityDetails.Text = ...
        'Blur: --   |   Brightness: --   |   FOV: --';

    severityValue.Text = 'Waiting for analysis';
    severityValue.FontColor = TEXT;

    confidenceValue.Text = '--';
    confidenceValue.FontColor = BLUE;

    gradeValue.Text = 'Grade: -- / 4';

    cla(probAxes);
    title(probAxes,'');

    cla(lesionAxes);
    title(lesionAxes,'');

    localizationStatus.Value = { ...
        'AI feature localization'
        ''
        'No analysis available.'
        ''
        'Upload an image to begin.'
        };

    statusLabel.Text = '● SYSTEM READY';
    statusLabel.FontColor = [0.55 0.95 0.65];

    analyzeButton.Enable = 'off';
    selectButton.Enable = 'on';
    saveButton.Enable = 'off';

end

%% ============================================================
% SAVE REPORT
% ============================================================

function saveReport(~,~)

    if isempty(lastResult)

        uialert(fig, ...
            'Run AI analysis before saving a report.', ...
            'No Analysis');

        return;

    end

    [file,path] = uiputfile( ...
        {'*.pdf','SeeBeyond Screening Report (*.pdf)'}, ...
        'Save SeeBeyond Screening Report', ...
        'SeeBeyond_Screening_Report.pdf');

    if isequal(file,0)
        return;
    end

    [~,~,ext] = fileparts(file);
    if isempty(ext)
        file = [file '.pdf'];
    end

    fullPath = fullfile(path,file);

    try

        generateSeeBeyondPDF(lastResult, ...
            char(selectedImagePath), ...
            fullPath);

        uialert(fig, ...
            sprintf('PDF report saved successfully:\n%s',fullPath), ...
            'Report Saved');

    catch ME

        uialert(fig, ...
            ME.message, ...
            'PDF Report Error');

    end

end

end
