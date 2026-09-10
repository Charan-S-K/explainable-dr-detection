function results = runSegmentationPipeline(inputImage)
% SeeBeyond - Final Stage 2 Retinal Segmentation Pipeline
%
% Segments:
%   1. Blood vessels
%   2. Hard exudates
%   3. Haemorrhages
%   4. Microaneurysms
%   5. Soft exudates
%
% Usage:
%
%   results = runSegmentationPipeline(I);
%
% OR
%
%   results = runSegmentationPipeline("image.jpg");


%% =========================================================
% Project root
%% =========================================================

thisFile = mfilename('fullpath');

segmentationDir = fileparts(thisFile);
srcDir = fileparts(segmentationDir);
projectRoot = fileparts(srcDir);

addpath(genpath(fullfile(projectRoot,'src')));


%% =========================================================
% Read input image
%% =========================================================

if ischar(inputImage) || isstring(inputImage)

    I = imread(inputImage);

else

    I = inputImage;

end


if size(I,3) == 1

    I = repmat(I,[1 1 3]);

end


fprintf('\n========================================\n');
fprintf('SEEBeyond STAGE-2 SEGMENTATION\n');
fprintf('========================================\n');

fprintf('Image size : %d x %d\n', ...
    size(I,1),size(I,2));


%% =========================================================
% Load all models only once
%% =========================================================

persistent vesselNetWeighted
persistent hardExudateNetV2
persistent haemorrhagePatchNetV4
persistent microaneurysmPatchNet
persistent softExudatePatchNet


if isempty(vesselNetWeighted)

    fprintf('\nLoading segmentation models...\n');

    S = load(fullfile(projectRoot, ...
        'models','vesselNetWeighted.mat'), ...
        'vesselNetWeighted');

    vesselNetWeighted = S.vesselNetWeighted;


    S = load(fullfile(projectRoot, ...
        'models','hardExudateNetV2.mat'), ...
        'hardExudateNetV2');

    hardExudateNetV2 = S.hardExudateNetV2;


    S = load(fullfile(projectRoot, ...
    'models','haemorrhagePatchNetV4.mat'), ...
    'haemorrhagePatchNetV4');

haemorrhagePatchNetV4 = ...
    S.haemorrhagePatchNetV4;


    S = load(fullfile(projectRoot, ...
        'models','microaneurysmPatchNet.mat'), ...
        'microaneurysmPatchNet');

    microaneurysmPatchNet = ...
        S.microaneurysmPatchNet;


   S = load(fullfile(projectRoot, ...
    'models','softExudatePatchNetV2.mat'), ...
    'softExudatePatchNetV2');

softExudatePatchNet = S.softExudatePatchNetV2;

    fprintf('All segmentation models loaded.\n');

end


%% =========================================================
% Retina mask
%% =========================================================

retinaMask = createRetinaMask(I);


%% =========================================================
% 1. BLOOD VESSEL SEGMENTATION
%% =========================================================

fprintf('\n[1/5] Segmenting blood vessels...\n');

vesselProb = predictResizedProbMap( ...
    I, ...
    vesselNetWeighted, ...
    [256 256]);

vesselMask = vesselProb >= 0.50;

vesselMask = vesselMask & retinaMask;

vesselMask = bwareaopen(vesselMask,2);

fprintf('Blood vessel segmentation complete.\n');


%% =========================================================
% 2. HARD EXUDATE SEGMENTATION
%
% Accepted model:
% hardExudateNetV2 + postprocessing
%% =========================================================

fprintf('\n[2/5] Segmenting hard exudates...\n');

hardExudateProb = predictPaddedProbMap( ...
    I, ...
    hardExudateNetV2, ...
    512);

hardExudateMask = hardExudateProb >= 0.50;

hardExudateMask = hardExudateMask & retinaMask;

hardExudateMask = ...
    bwareaopen(hardExudateMask,8);

% Slight erosion used in accepted V2 cleanup
hardExudateMask = imerode( ...
    hardExudateMask, ...
    strel('disk',1,0));

fprintf('Hard Exudate segmentation complete.\n');


%% =========================================================
% 3. HAEMORRHAGE SEGMENTATION
%
% Improved model = Haemorrhage Patch V4
% Validated threshold = 0.45
%% =========================================================

fprintf('\n[3/5] Segmenting haemorrhages...\n');

haemorrhageProb = ...
    predictHaemorrhageProbMapV2( ...
        I, ...
        haemorrhagePatchNetV4);

haemorrhageMask = ...
    haemorrhageProb >= 0.45;

haemorrhageMask = ...
    haemorrhageMask & retinaMask;

haemorrhageMask = ...
    bwareaopen(haemorrhageMask,6);

fprintf('Haemorrhage segmentation complete.\n');


%% =========================================================
% 4. MICROANEURYSM SEGMENTATION
%
% Validated threshold = 0.50
%% =========================================================

fprintf('\n[4/5] Segmenting microaneurysms...\n');

microaneurysmProb = ...
    predictMicroaneurysmProbMap( ...
        I, ...
        microaneurysmPatchNet);

microaneurysmMask = ...
    microaneurysmProb >= 0.50;

microaneurysmMask = ...
    microaneurysmMask & retinaMask;

% Do NOT aggressively remove small regions.
% Real microaneurysms are extremely small.

fprintf('Microaneurysm segmentation complete.\n');


%% =========================================================
% 5. SOFT EXUDATE SEGMENTATION
%
% Calibrated threshold = 0.20
%% =========================================================

fprintf('\n[5/5] Segmenting soft exudates...\n');

softExudateProb = ...
    predictSoftExudateProbMap( ...
        I, ...
        softExudatePatchNet);

softExudateMask = ...
    softExudateProb >= 0.20;

softExudateMask = ...
    softExudateMask & retinaMask;

softExudateMask = ...
    bwareaopen(softExudateMask,3);

fprintf('Soft Exudate segmentation complete.\n');


%% =========================================================
% Combined visual overlay
%% =========================================================

overlay = createSegmentationOverlay( ...
    I, ...
    vesselMask, ...
    hardExudateMask, ...
    haemorrhageMask, ...
    microaneurysmMask, ...
    softExudateMask);


%% =========================================================
% Pixel statistics
%% =========================================================

retinaPixels = nnz(retinaMask);

if retinaPixels == 0
    retinaPixels = 1;
end


structure = [
    "Blood Vessel"
    "Hard Exudate"
    "Haemorrhage"
    "Microaneurysm"
    "Soft Exudate"
    ];


pixelCount = [
    nnz(vesselMask)
    nnz(hardExudateMask)
    nnz(haemorrhageMask)
    nnz(microaneurysmMask)
    nnz(softExudateMask)
    ];


retinaPercent = ...
    100 .* pixelCount ./ retinaPixels;


summaryTable = table( ...
    structure, ...
    pixelCount, ...
    retinaPercent, ...
    'VariableNames', ...
    {'Structure','PixelCount','RetinaPercent'});


%% =========================================================
% Return everything
%% =========================================================

results = struct;

results.retinaMask = retinaMask;

results.vesselMask = vesselMask;

results.hardExudateMask = ...
    hardExudateMask;

results.haemorrhageMask = ...
    haemorrhageMask;

results.microaneurysmMask = ...
    microaneurysmMask;

results.softExudateMask = ...
    softExudateMask;

results.overlay = overlay;

results.summary = summaryTable;


%% Threshold information

results.thresholds = struct( ...
    'vessel',0.50, ...
    'hardExudate',0.50, ...
    'haemorrhage',0.45, ...
    'microaneurysm',0.50, ...
    'softExudate',0.20);


fprintf('\n========================================\n');
fprintf('STAGE-2 SEGMENTATION COMPLETE\n');
fprintf('========================================\n');

disp(summaryTable)

end



%% =========================================================
% LOCAL FUNCTION
% Retina field extraction
%% =========================================================

function retinaMask = createRetinaMask(I)

G = im2double(rgb2gray(I));

retinaMask = G > 0.03;

if any(retinaMask(:))

    retinaMask = ...
        bwareafilt(retinaMask,1);

    retinaMask = ...
        imfill(retinaMask,'holes');

end

end



%% =========================================================
% LOCAL FUNCTION
% Standard resized network inference
%
% Used for DRIVE vessel model
%% =========================================================

function probMap = predictResizedProbMap( ...
    I,net,inputSize)

originalH = size(I,1);
originalW = size(I,2);

Iresize = imresize(I,inputSize);

scores = getNetworkScores(net,Iresize);

probSmall = scores(:,:,2);

probMap = imresize( ...
    probSmall, ...
    [originalH originalW], ...
    'bilinear');

end



%% =========================================================
% LOCAL FUNCTION
% Aspect-ratio preserving square prediction
%
% Used by 512 x 512 whole-retina models.
%% =========================================================

function probMap = predictPaddedProbMap( ...
    I,net,targetSize)

originalH = size(I,1);
originalW = size(I,2);

scale = min( ...
    targetSize/originalH, ...
    targetSize/originalW);

newH = max(1,round(originalH*scale));
newW = max(1,round(originalW*scale));

Iresize = imresize(I,[newH newW]);

padTop = floor((targetSize-newH)/2);
padLeft = floor((targetSize-newW)/2);

Ipad = zeros( ...
    targetSize, ...
    targetSize, ...
    3, ...
    'like',I);

r1 = padTop + 1;
r2 = padTop + newH;

c1 = padLeft + 1;
c2 = padLeft + newW;

Ipad(r1:r2,c1:c2,:) = Iresize;

scores = getNetworkScores(net,Ipad);

probPadded = scores(:,:,2);

probCrop = ...
    probPadded(r1:r2,c1:c2);

probMap = imresize( ...
    probCrop, ...
    [originalH originalW], ...
    'bilinear');

end



%% =========================================================
% LOCAL FUNCTION
% dlnetwork prediction
%% =========================================================

function scores = getNetworkScores(net,I)

if size(I,3) == 1
    I = repmat(I,[1 1 3]);
end

X = reshape( ...
    I, ...
    size(I,1), ...
    size(I,2), ...
    3, ...
    1);

scores = minibatchpredict( ...
    net, ...
    X, ...
    MiniBatchSize=1, ...
    ExecutionEnvironment="gpu");

if isa(scores,'dlarray')

    scores = extractdata(scores);

end

if isa(scores,'gpuArray')

    scores = gather(scores);

end

if ndims(scores) == 4

    scores = scores(:,:,:,1);

end

if size(scores,3) < 2

    error( ...
        'Segmentation network did not return two classes.');

end

end



%% =========================================================
% LOCAL FUNCTION
% Combined visualization
%% =========================================================

function overlay = createSegmentationOverlay( ...
    I, ...
    vesselMask, ...
    hardExudateMask, ...
    haemorrhageMask, ...
    microaneurysmMask, ...
    softExudateMask)

overlay = im2double(I);

alpha = 0.55;

% Vessel
overlay = blendMask( ...
    overlay,vesselMask,[0 1 1],alpha);

% Hard Exudate
overlay = blendMask( ...
    overlay,hardExudateMask,[1 1 0],alpha);

% Haemorrhage
overlay = blendMask( ...
    overlay,haemorrhageMask,[1 0 0],alpha);

% Microaneurysm
overlay = blendMask( ...
    overlay,microaneurysmMask,[1 0 1],alpha);

% Soft Exudate
overlay = blendMask( ...
    overlay,softExudateMask,[0 0.4 1],alpha);

overlay = min(max(overlay,0),1);

end



%% =========================================================
% LOCAL FUNCTION
% Overlay one binary mask
%% =========================================================

function imageOut = blendMask( ...
    imageIn,mask,color,alpha)

imageOut = imageIn;

for channel = 1:3

    currentChannel = imageOut(:,:,channel);

    currentChannel(mask) = ...
        (1-alpha).*currentChannel(mask) + ...
        alpha.*color(channel);

    imageOut(:,:,channel) = ...
        currentChannel;

end

end