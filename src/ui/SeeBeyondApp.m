function SeeBeyondApp
% SeeBeyondApp
% SeeBeyond final hackathon UI.
%
% Backend:
%   runSeeBeyond()
%
% The validated inference pipeline is not modified.

%% =========================================================
% PROJECT SETUP
% ==========================================================

projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));

addpath(genpath(fullfile(projectRoot,'src')));
rehash;

selectedImagePath = "";
lastResult = [];

%% =========================================================
% COLOR THEME
% ==========================================================

BG          = [0.95 0.97 0.98];
WHITE       = [1.00 1.00 1.00];
NAVY        = [0.08 0.13 0.20];
TEXT        = [0.10 0.14 0.20];
SECONDARY   = [0.35 0.40 0.47];
BLUE        = [0.10 0.42 0.75];
GREEN       = [0.10 0.55 0.30];
ORANGE      = [0.85 0.52 0.08];
RED         = [0.75 0.18 0.18];
BORDER      = [0.84 0.87 0.90];

%% =========================================================
% MAIN WINDOW
% ==========================================================

fig = uifigure( ...
    'Name','SeeBeyond - AI Retinal Screening', ...
    'Position',[60 40 1500 900], ...
    'Color',BG);

%% =========================================================
% HEADER
% ==========================================================

header = uipanel(fig, ...
    'Position',[0 825 1500 75], ...
    'BorderType','none', ...
    'BackgroundColor',NAVY);

uilabel(header, ...
    'Position',[30 30 300 32], ...
    'Text','SEEBEYOND', ...
    'FontSize',25, ...
    'FontWeight','bold', ...
    'FontColor',WHITE);

uilabel(header, ...
    'Position',[32 8 600 22], ...
    'Text','AI-Assisted Retinal Screening Platform', ...
    'FontSize',12, ...
    'FontColor',[0.82 0.87 0.92]);

statusLabel = uilabel(header, ...
    'Position',[1080 24 380 30], ...
    'Text','● SYSTEM READY', ...
    'HorizontalAlignment','right', ...
    'FontSize',13, ...
    'FontWeight','bold', ...
    'FontColor',[0.55 0.95 0.65]);

%% =========================================================
% IMAGE CARD
% ==========================================================

imageCard = uipanel(fig, ...
    'Position',[25 455 650 345], ...
    'BackgroundColor',WHITE, ...
    'BorderType','line', ...
    'HighlightColor',BORDER);

uilabel(imageCard, ...
    'Position',[22 305 400 25], ...
    'Text','RETINAL IMAGE', ...
    'FontSize',13, ...
    'FontWeight','bold', ...
    'FontColor',TEXT);

uilabel(imageCard, ...
    'Position',[22 280 500 20], ...
    'Text','Upload a fundus photograph for AI-assisted screening', ...
    'FontSize',10, ...
    'FontColor',SECONDARY);

imageAxes = uiaxes(imageCard, ...
    'Position',[22 45 606 225], ...
    'Color',WHITE);

imageAxes.XTick = [];
imageAxes.YTick = [];
imageAxes.Box = 'on';
imageAxes.Toolbar.Visible = 'off';

title(imageAxes,'No image selected', ...
    'Color',SECONDARY);

imageInfoLabel = uilabel(imageCard, ...
    'Position',[22 15 606 22], ...
    'Text','No image selected', ...
    'HorizontalAlignment','center', ...
    'FontSize',10, ...
    'FontColor',SECONDARY);

%% =========================================================
% QUALITY CARD
% ==========================================================

qualityCard = uipanel(fig, ...
    'Position',[700 675 775 125], ...
    'BackgroundColor',WHITE, ...
    'BorderType','line', ...
    'HighlightColor',BORDER);

uilabel(qualityCard, ...
    'Position',[22 80 220 22], ...
    'Text','IMAGE QUALITY', ...
    'FontSize',11, ...
    'FontWeight','bold', ...
    'FontColor',SECONDARY);

qualityValue = uilabel(qualityCard, ...
    'Position',[22 42 300 32], ...
    'Text','Waiting for image', ...
    'FontSize',19, ...
    'FontWeight','bold', ...
    'FontColor',TEXT);

qualityDetails = uilabel(qualityCard, ...
    'Position',[350 42 390 30], ...
    'Text','Blur: --   Brightness: --   FOV: --', ...
    'FontSize',10, ...
    'FontColor',SECONDARY);

%% =========================================================
% DR RESULT CARD
% ==========================================================

drCard = uipanel(fig, ...
    'Position',[700 455 775 200], ...
    'BackgroundColor',WHITE, ...
    'BorderType','line', ...
    'HighlightColor',BORDER);

uilabel(drCard, ...
    'Position',[22 160 300 22], ...
    'Text','PRIMARY AI ASSESSMENT', ...
    'FontSize',11, ...
    'FontWeight','bold', ...
    'FontColor',SECONDARY);

severityTitle = uilabel(drCard, ...
    'Position',[22 125 180 20], ...
    'Text','DR SEVERITY', ...
    'FontSize',10, ...
    'FontWeight','bold', ...
    'FontColor',SECONDARY);

severityValue = uilabel(drCard, ...
    'Position',[22 82 360 42], ...
    'Text','Waiting for analysis', ...
    'FontSize',22, ...
    'FontWeight','bold', ...
    'FontColor',TEXT);

gradeValue = uilabel(drCard, ...
    'Position',[22 48 220 25], ...
    'Text','DR Grade: -- / 4', ...
    'FontSize',12, ...
    'FontColor',SECONDARY);

confidenceTitle = uilabel(drCard, ...
    'Position',[440 125 200 20], ...
    'Text','CONFIDENCE', ...
    'FontSize',10, ...
    'FontWeight','bold', ...
    'FontColor',SECONDARY);

confidenceValue = uilabel(drCard, ...
    'Position',[440 82 250 42], ...
    'Text','--', ...
    'FontSize',26, ...
    'FontWeight','bold', ...
    'FontColor',BLUE);

confidenceInfo = uilabel(drCard, ...
    'Position',[440 48 300 25], ...
    'Text','AI classification confidence', ...
    'FontSize',10, ...
    'FontColor',SECONDARY);

%% =========================================================
% PROBABILITY CARD
% ==========================================================

probCard = uipanel(fig, ...
    'Position',[25 185 650 250], ...
    'BackgroundColor',WHITE, ...
    'BorderType','line', ...
    'HighlightColor',BORDER);

uilabel(probCard, ...
    'Position',[22 215 350 22], ...
    'Text','DR CLASSIFICATION PROBABILITY', ...
    'FontSize',11, ...
    'FontWeight','bold', ...
    'FontColor',TEXT);

probAxes = uiaxes(probCard, ...
    'Position',[22 25 606 175], ...
    'Color',WHITE);

probAxes.Box = 'on';
probAxes.Toolbar.Visible = 'off';

%% =========================================================
% LOCALIZATION CARD
% ==========================================================

lesionCard = uipanel(fig, ...
    'Position',[700 185 775 250], ...
    'BackgroundColor',WHITE, ...
    'BorderType','line', ...
    'HighlightColor',BORDER);

uilabel(lesionCard, ...
    'Position',[22 215 350 22], ...
    'Text','AI LESION LOCALIZATION', ...
    'FontSize',11, ...
    'FontWeight','bold', ...
    'FontColor',TEXT);

lesionAxes = uiaxes(lesionCard, ...
    'Position',[22 25 450 175], ...
    'Color',WHITE);

lesionAxes.Box = 'on';
lesionAxes.Toolbar.Visible = 'off';

localizationStatus = uitextarea(lesionCard, ...
    'Position',[490 30 260 165], ...
    'Editable','off', ...
    'FontSize',10, ...
    'FontColor',TEXT, ...
    'BackgroundColor',WHITE, ...
    'Value',{ ...
    'Localization status'
    ''
    'No analysis available.'
    });

%% =========================================================
% BOTTOM CONTROL BAR
% ==========================================================

controlPanel = uipanel(fig, ...
    'Position',[25 105 1450 60], ...
    'BackgroundColor',WHITE, ...
    'BorderType','line', ...
    'HighlightColor',BORDER);

selectButton = uibutton(controlPanel, ...
    'push', ...
    'Position',[20 12 190 36], ...
    'Text','UPLOAD FUNDUS IMAGE', ...
    'FontSize',11, ...
    'FontWeight','bold', ...
    'FontColor',WHITE, ...
    'BackgroundColor',BLUE, ...
    'ButtonPushedFcn',@selectImage);

analyzeButton = uibutton(controlPanel, ...
    'push', ...
    'Position',[225 12 160 36], ...
    'Text','RUN AI ANALYSIS', ...
    'FontSize',11, ...
    'FontWeight','bold', ...
    'FontColor',WHITE, ...
    'BackgroundColor',GREEN, ...
    'Enable','off', ...
    'ButtonPushedFcn',@analyzeImage);

clearButton = uibutton(controlPanel, ...
    'push', ...
    'Position',[400 12 110 36], ...
    'Text','CLEAR', ...
    'FontSize',11, ...
    'FontColor',TEXT, ...
    'BackgroundColor',[0.90 0.92 0.94], ...
    'ButtonPushedFcn',@clearResults);

saveButton = uibutton(controlPanel, ...
    'push', ...
    'Position',[525 12 145 36], ...
    'Text','SAVE REPORT', ...
    'FontSize',11, ...
    'FontWeight','bold', ...
    'FontColor',TEXT, ...
    'BackgroundColor',[0.90 0.92 0.94], ...
    'Enable','off', ...
    'ButtonPushedFcn',@saveReport);

uilabel(controlPanel, ...
    'Position',[850 12 570 36], ...
    'Text','AI-assisted screening prototype  |  Not a clinical diagnosis', ...
    'HorizontalAlignment','right', ...
    'FontSize',10, ...
    'FontColor',SECONDARY);

%% =========================================================
% FOOTER
% ==========================================================

footer = uipanel(fig, ...
    'Position',[25 20 1450 65], ...
    'BorderType','none', ...
    'BackgroundColor',BG);

uilabel(footer, ...
    'Position',[0 35 1000 22], ...
    'Text','SeeBeyond  •  Quality Assessment  •  DR Classification  •  Lesion Localization', ...
    'FontSize',10, ...
    'FontColor',SECONDARY);

uilabel(footer, ...
    'Position',[0 8 1400 22], ...
    'Text','Segmentation outputs are AI localization/explainability signals and are not independent clinical diagnoses.', ...
    'FontSize',9, ...
    'FontColor',SECONDARY);

%% =========================================================
% SELECT IMAGE
% ==========================================================

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

            title(imageAxes,'Original Fundus Image', ...
                'Color',TEXT);

            imageInfoLabel.Text = sprintf( ...
                '%s   |   %d × %d pixels', ...
                file,size(I,2),size(I,1));

            statusLabel.Text = '● IMAGE READY';
            statusLabel.FontColor = [0.95 0.65 0.10];

            analyzeButton.Enable = 'on';
            saveButton.Enable = 'off';

            qualityValue.Text = 'Ready for analysis';
            qualityValue.FontColor = TEXT;

            severityValue.Text = 'Waiting for analysis';
            confidenceValue.Text = '--';
            gradeValue.Text = 'DR Grade: -- / 4';

            localizationStatus.Value = { ...
                'Localization status'
                ''
                'Run AI analysis to view'
                'lesion localization.'
                };

            reportTextReset();

        catch ME

            uialert(fig,ME.message,'Image Loading Error');

        end

    end

%% =========================================================
% ANALYZE IMAGE
% ==========================================================

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

        % =====================================================
        % RUN THE VALIDATED SEEBEYOND BACKEND
        % =====================================================

        result = runSeeBeyond(char(selectedImagePath));

        lastResult = result;

        % =====================================================
        % IMAGE DISPLAY
        % =====================================================

        if isfield(result,'segmentationAvailable') && ...
                result.segmentationAvailable && ...
                isfield(result.segmentation,'overlay')

            imshow(result.segmentation.overlay, ...
                'Parent',imageAxes);

            title(imageAxes, ...
                'AI Lesion Localization Overlay', ...
                'Color',TEXT);

        else

            imshow(result.image, ...
                'Parent',imageAxes);

            title(imageAxes, ...
                'Original Fundus Image', ...
                'Color',TEXT);

        end

        % =====================================================
        % IMAGE QUALITY
        % =====================================================

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

        % =====================================================
        % DR CLASSIFICATION
        % =====================================================

        severityValue.Text = char(result.severity);

        confidenceValue.Text = sprintf( ...
            '%.2f%%', ...
            result.confidencePercent);

        gradeValue.Text = sprintf( ...
            'DR Grade: %d / 4', ...
            result.drGrade);

        % =====================================================
        % DR PROBABILITY CHART
        % =====================================================

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

        probAxes.YLim = [0 100];

        ylabel(probAxes,'Probability (%)', ...
            'Color',TEXT);

        title(probAxes, ...
            'DR Classification Probability', ...
            'Color',TEXT);

        probAxes.XColor = TEXT;
        probAxes.YColor = TEXT;

        grid(probAxes,'on');

               % =====================================================
        % GET SEGMENTATION DATA
        % =====================================================

        seg = result.segmentation;

        % Actual segmentation masks from the validated backend
        masks = {
            seg.vesselMask
            seg.hardExudateMask
            seg.haemorrhageMask
            seg.microaneurysmMask
            seg.softExudateMask
            };

        lesionNames = { ...
            'Blood Vessel'
            'Hard Exudate'
            'Haemorrhage'
            'Microaneurysm'
            'Soft Exudate'
            };

        % Calculate actual pixel counts
        counts = zeros(5,1);

        for k = 1:5
            counts(k) = nnz(masks{k});
        end

        % Calculate percentage of retinal area
        retinaArea = nnz(seg.retinaMask);

        retinaPercent = zeros(5,1);

        if retinaArea > 0
            for k = 1:5
                retinaPercent(k) = ...
                    100 * counts(k) / retinaArea;
            end
        end

        % A mask is considered AI-localized when it contains
        % at least one segmented pixel.
        aiLocalized = counts > 0;


        % =====================================================
        % LESION LOCALIZATION CHART
        % =====================================================

        cla(lesionAxes);

        bar(lesionAxes,counts);

        lesionAxes.XTick = 1:5;

        lesionAxes.XTickLabel = { ...
            'Vessel'
            'Hard EX'
            'Haemorrhage'
            'MA'
            'Soft EX'};

        lesionAxes.XTickLabelRotation = 25;

        ylabel(lesionAxes,'Segmented Pixels', ...
            'Color',TEXT);

        title(lesionAxes, ...
            'AI Localization by Segmented Pixel Count', ...
            'Color',TEXT);

        lesionAxes.XColor = TEXT;
        lesionAxes.YColor = TEXT;

        grid(lesionAxes,'on');


        % =====================================================
        % LOCALIZATION TEXT
        % =====================================================

        localizationLines = { ...
            'LOCALIZATION SUMMARY'
            ''
            sprintf('Retinal area: %d pixels',retinaArea)
            ''};

        for k = 1:numel(lesionNames)

            if aiLocalized(k)
                state = 'AI-localized';
            else
                state = 'Not localized';
            end

            localizationLines{end+1} = sprintf( ...
                '%-16s %s', ...
                lesionNames{k}, ...
                state);

            localizationLines{end+1} = sprintf( ...
                '  %d pixels | %.3f%% retina', ...
                counts(k), ...
                retinaPercent(k));

        end

        localizationLines{end+1} = '';
        localizationLines{end+1} = ...
            'Localization is an AI explainability signal.';

        localizationLines{end+1} = ...
            'It is not an independent clinical diagnosis.';

        localizationStatus.Value = localizationLines;

        % =====================================================
        % REPORT
        % =====================================================

        reportLines = {};

        reportLines{end+1} = ...
            'SEEBEYOND AI SCREENING REPORT';

        reportLines{end+1} = ...
            '============================================================';

        reportLines{end+1} = '';

        reportLines{end+1} = ...
            ['Image: ' char(selectedImagePath)];

        reportLines{end+1} = '';

        reportLines{end+1} = 'IMAGE QUALITY';

        reportLines{end+1} = ...
            ['Status       : ' char(result.qualityStatus)];

        if isfield(result,'qualityReport') && ...
                ~isempty(fieldnames(result.qualityReport))

            qr = result.qualityReport;

            reportLines{end+1} = sprintf( ...
                'Blur score   : %.6f',qr.blurScore);

            reportLines{end+1} = sprintf( ...
                'Brightness   : %.4f',qr.brightnessScore);

            reportLines{end+1} = sprintf( ...
                'FOV score    : %.4f',qr.fovScore);

        end

        reportLines{end+1} = '';

        reportLines{end+1} = 'DR ASSESSMENT';

        reportLines{end+1} = ...
            ['Severity     : ' char(result.severity)];

        reportLines{end+1} = sprintf( ...
            'DR Grade     : %d / 4',result.drGrade);

        reportLines{end+1} = sprintf( ...
            'Confidence   : %.2f%%',result.confidencePercent);

        reportLines{end+1} = '';

        reportLines{end+1} = 'CLASS PROBABILITIES';

        for k = 1:numel(result.classNames)

            reportLines{end+1} = sprintf( ...
                '  %-20s %.2f%%', ...
                char(result.classNames(k)), ...
                result.classScores(k)*100);

        end

        reportLines{end+1} = '';

        reportLines{end+1} = 'AI LESION LOCALIZATION';

        for k = 1:numel(lesionNames)

            if k <= numel(counts) && ...
                    k <= numel(retinaPercent)

                reportLines{end+1} = sprintf( ...
                    '  %-18s %d pixels (%.3f%% retina)', ...
                    char(lesionNames{k}), ...
                    counts(k), ...
                    retinaPercent(k));

            else

                reportLines{end+1} = sprintf( ...
                    '  %-18s AI-localized', ...
                    char(lesionNames{k}));

            end

        end

        reportLines{end+1} = '';

        reportLines{end+1} = ...
            '============================================================';

        reportLines{end+1} = ...
            'PROTOTYPE DISCLAIMER';

        reportLines{end+1} = ...
            'DR classification is the primary AI severity result.';

        reportLines{end+1} = ...
            'Segmentation outputs are AI localization/explainability';

        reportLines{end+1} = ...
            'signals and are NOT independent clinical diagnoses.';

        reportLines{end+1} = ...
            'This system does not replace professional eye care.';

        reportLines{end+1} = ...
            '============================================================';

        setappdata(fig,'reportLines',reportLines);

        % =====================================================
        % COMPLETE
        % =====================================================

        statusLabel.Text = '● ANALYSIS COMPLETE';
        statusLabel.FontColor = GREEN;

        saveButton.Enable = 'on';

    catch ME

        statusLabel.Text = '● ANALYSIS FAILED';
        statusLabel.FontColor = RED;

        uialert(fig, ...
            ME.message, ...
            'SeeBeyond Analysis Error');

    end

        analyzeButton.Enable = 'on';
    selectButton.Enable = 'on';

    end


    %% =========================================================
    % CLEAR RESULTS
    % ==========================================================
    function clearResults(~,~)

        selectedImagePath = "";
        lastResult = [];

        % Clear retinal image
        cla(imageAxes);
        title(imageAxes,'No image selected','Color',SECONDARY);

        imageInfoLabel.Text = 'No image selected';

        % Reset image quality
        qualityValue.Text = 'Waiting for image';
        qualityValue.FontColor = TEXT;

        qualityDetails.Text = ...
            'Blur: --   |   Brightness: --   |   FOV: --';

        % Clear probability chart
        cla(probAxes);
        title(probAxes,'DR Classification Probability','Color',TEXT);
        probAxes.XTick = [];
        probAxes.YTick = [];

        % Reset DR result
        severityValue.Text = 'Waiting for analysis';
        confidenceValue.Text = '--';
        gradeValue.Text = 'DR Grade: -- / 4';

        % Clear localization chart
        cla(lesionAxes);
        title(lesionAxes,'Localization Data','Color',TEXT);
        lesionAxes.XTick = [];
        lesionAxes.YTick = [];

        % Reset localization text
        localizationStatus.Value = { ...
            'Localization status'
            ''
            'No analysis available.'
            };

        % Reset report
        reportTextReset();

        % Reset buttons
        analyzeButton.Enable = 'off';
        saveButton.Enable = 'off';
        selectButton.Enable = 'on';

        % Reset system status
        statusLabel.Text = '● SYSTEM READY';
        statusLabel.FontColor = [0.55 0.95 0.65];

    end

    %% =========================================================
    % RESET REPORT
    % ==========================================================

    function reportTextReset()

        setappdata(fig,'reportLines',{ ...
            'SEEBEYOND AI SCREENING REPORT'
            '============================================================'
            ''
            'No analysis available.'
            ''
            'Upload a fundus image and run AI analysis.'
            });

    end


    %% =========================================================
    % SAVE REPORT
    % ==========================================================

    function saveReport(~,~)

        if ~isappdata(fig,'reportLines')

            uialert(fig, ...
                'No report is available to save.', ...
                'Save Report');

            return;

        end

        reportLines = getappdata(fig,'reportLines');

        [file,path] = uiputfile( ...
            {'*.txt','Text Report (*.txt)'}, ...
            'Save SeeBeyond AI Screening Report', ...
            'SeeBeyond_AI_Report.txt');

        if isequal(file,0)
            return;
        end

        reportPath = fullfile(path,file);

        fid = fopen(reportPath,'w');

        if fid == -1

            uialert(fig, ...
                'Could not create the report file.', ...
                'Save Report Error');

            return;

        end

        for k = 1:numel(reportLines)

            fprintf(fid,'%s\n',reportLines{k});

        end

        fclose(fid);

        uialert(fig, ...
            sprintf('Report saved successfully:\n\n%s',reportPath), ...
            'Report Saved');

    end


end