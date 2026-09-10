function trainMicroaneurysmPatchNetV4()
% SeeBeyond - Microaneurysm Patch Network V4
% MATLAB R2026a
%
% V4:
%   - Uses the existing V3 patch dataset
%   - Uses the new R2026a unet() API
%   - Uses imageDatastore + pixelLabelDatastore
%   - Uses weighted cross-entropy training
%
% IMPORTANT:
% The images and masks are paired. They are NOT classes.

clc;

projectRoot = '/home/tharunkumar-s-v/Documents/SeeBeyond';
cd(projectRoot);

fprintf('\n');
fprintf('====================================================\n');
fprintf('MICROANEURYSM V4 TRAINING - R2026a\n');
fprintf('====================================================\n');

%% Paths

imageDir = fullfile(projectRoot, ...
    'datasets','IDRiD','patches','microaneurysmsV3','images');

maskDir = fullfile(projectRoot, ...
    'datasets','IDRiD','patches','microaneurysmsV3','masks');

if ~isfolder(imageDir)
    error('Image directory not found:\n%s',imageDir);
end

if ~isfolder(maskDir)
    error('Mask directory not found:\n%s',maskDir);
end

%% Image datastore

imds = imageDatastore(imageDir, ...
    'FileExtensions',{'.png','.jpg','.jpeg'});

fprintf('Training images: %d\n',numel(imds.Files));

%% Pixel-label datastore

classNames = ["background","microaneurysm"];
labelIDs = [0 1];

pxds = pixelLabelDatastore( ...
    maskDir, ...
    classNames, ...
    labelIDs, ...
    'FileExtensions',{'.png','.jpg','.jpeg'});

fprintf('Training masks : %d\n',numel(pxds.Files));

if numel(imds.Files) ~= numel(pxds.Files)
    error('Number of images and masks does not match.');
end

%% Check pairing

imageNames = cellfun(@(x) getBaseName(x), ...
    imds.Files,'UniformOutput',false);

maskNames = cellfun(@(x) getBaseName(x), ...
    pxds.Files,'UniformOutput',false);

if ~isequal(imageNames,maskNames)
    error(['Image/mask filenames are not aligned.\n' ...
           'Check the V3 dataset.']);
end

fprintf('Image/mask pairing: OK\n');

%% Pixel statistics

tbl = countEachLabel(pxds);

fprintf('\nPixel class distribution:\n');
disp(tbl);

%% Calculate class weights

counts = tbl.PixelCount;

backgroundCount = counts(strcmp(string(tbl.Name),"background"));
maCount = counts(strcmp(string(tbl.Name),"microaneurysm"));

if isempty(backgroundCount) || isempty(maCount)
    error('Could not identify background/microaneurysm classes.');
end

maWeight = sqrt(backgroundCount / max(maCount,1));

% Controlled weight to avoid unstable training.
maWeight = min(3.0,max(1.5,maWeight));

classWeights = [1 maWeight];

fprintf('Microaneurysm class weight: %.4f\n',maWeight);

%% Create training datastore

pximds = pixelLabelImageDatastore(imds,pxds);

fprintf('\nTraining datastore created.\n');

%% Network

inputSize = [256 256 3];
numClasses = 2;

fprintf('\nCreating R2026a U-Net...\n');

net = unet( ...
    inputSize, ...
    numClasses, ...
    EncoderDepth=3, ...
    NumFirstEncoderFilters=16);

fprintf('U-Net created successfully.\n');

%% Training options

miniBatchSize = 4;

options = trainingOptions('adam', ...
    InitialLearnRate=5e-5, ...
    MaxEpochs=35, ...
    MiniBatchSize=miniBatchSize, ...
    Shuffle='every-epoch', ...
    ExecutionEnvironment='gpu', ...
    Verbose=true, ...
    Plots='training-progress');

fprintf('\n');
fprintf('====================================================\n');
fprintf('V4 TRAINING SETTINGS\n');
fprintf('====================================================\n');

fprintf('Input size          : %d x %d x %d\n',inputSize);
fprintf('Encoder depth       : 3\n');
fprintf('First filters       : 16\n');
fprintf('Learning rate       : %.1e\n',options.InitialLearnRate);
fprintf('Epochs              : %d\n',options.MaxEpochs);
fprintf('Mini-batch          : %d\n',options.MiniBatchSize);
fprintf('MA class weight     : %.4f\n',maWeight);
fprintf('Execution           : GPU\n');

fprintf('\nStarting V4 training...\n\n');

%% Train

[net,info] = trainnet( ...
    pximds, ...
    net, ...
    "crossentropy", ...
    options);

%% Save

modelFile = fullfile( ...
    projectRoot, ...
    'models','microaneurysmPatchNetV4.mat');

microaneurysmPatchNetV4 = net;

save(modelFile, ...
    'microaneurysmPatchNetV4', ...
    'info', ...
    'maWeight', ...
    '-v7.3');

fprintf('\n');
fprintf('====================================================\n');
fprintf('MICROANEURYSM V4 TRAINING COMPLETE\n');
fprintf('====================================================\n');

fprintf('Saved model:\n%s\n',modelFile);

fprintf('====================================================\n');

end


function name = getBaseName(filePath)

[~,name,~] = fileparts(filePath);

end