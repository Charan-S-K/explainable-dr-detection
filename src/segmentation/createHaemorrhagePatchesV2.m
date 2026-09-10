%% SeeBeyond - Haemorrhage V2 Patch Generator
% Accuracy improvement experiment
%
% Creates:
%   1. Haemorrhage-positive patches
%   2. Vessel hard-negative patches
%   3. Normal retinal negative patches
%
% IMPORTANT:
% Does NOT modify the old haemorrhage model.

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

rng(42);

fprintf('\n========================================\n');
fprintf('HAEMORRHAGE V2 PATCH GENERATION\n');
fprintf('========================================\n');

%% =========================================================
% Load SAME train / validation split
%% =========================================================

load('models/haemorrhageNet.mat', ...
    'trainHEIdx','valHEIdx');

fprintf('Training images   : %d\n',numel(trainHEIdx));
fprintf('Validation images : %d\n',numel(valHEIdx));

%% =========================================================
% Load vessel model
%% =========================================================

S = load('models/vesselNetWeighted.mat', ...
    'vesselNetWeighted');

vesselNetWeighted = S.vesselNetWeighted;

fprintf('Vessel model loaded.\n');

%% =========================================================
% Dataset paths
%% =========================================================

idridRoot = fullfile(pwd, ...
    'datasets','IDRiD','Segmentation','A. Segmentation');

imageRoot = fullfile(idridRoot, ...
    '1. Original Images','a. Training Set');

heMaskRoot = fullfile(pwd, ...
    'datasets','IDRiD','processed_masks','haemorrhages');

%% New V2 patch directory

patchRoot = fullfile(pwd, ...
    'datasets','IDRiD','patches','haemorrhages_v2');

imageOut = fullfile(patchRoot,'images');
maskOut  = fullfile(patchRoot,'masks');

%% Remove ONLY previous V2 patches

if exist(patchRoot,'dir')
    rmdir(patchRoot,'s');
end

mkdir(imageOut);
mkdir(maskOut);

%% =========================================================
% Image datastore
%% =========================================================

imds = imageDatastore(imageRoot);

patchSize = 256;

positivePerImage   = 40;
vesselNegPerImage  = 20;
normalNegPerImage  = 15;

counter = 0;

positiveCount = 0;
vesselNegCount = 0;
normalNegCount = 0;

%% =========================================================
% Generate patches from TRAINING images only
%% =========================================================

for n = 1:numel(trainHEIdx)

    idx = trainHEIdx(n);

    I = readimage(imds,idx);

    [~,imageID,~] = ...
        fileparts(imds.Files{idx});

    hePath = fullfile( ...
        heMaskRoot,[imageID '.png']);

    HE = imread(hePath) > 0;

    [h,w,~] = size(I);

    fprintf('\n[%d/%d] %s\n', ...
        n,numel(trainHEIdx),imageID);

    %% =====================================================
    % Retina mask
    %% =====================================================

    G = im2double(rgb2gray(I));

    retinaMask = G > 0.03;

    if any(retinaMask(:))

        retinaMask = ...
            bwareafilt(retinaMask,1);

        retinaMask = ...
            imfill(retinaMask,'holes');

    end

    %% =====================================================
    % Predict vessels
    %
    % DRIVE vessel model operates at 256x256.
    %% =====================================================

    I256 = imresize(I,[256 256]);

    X = reshape(I256,256,256,3,1);

    vesselScores = minibatchpredict( ...
        vesselNetWeighted, ...
        X, ...
        MiniBatchSize=1, ...
        ExecutionEnvironment="gpu");

    if isa(vesselScores,'dlarray')
        vesselScores = extractdata(vesselScores);
    end

    if isa(vesselScores,'gpuArray')
        vesselScores = gather(vesselScores);
    end

    if ndims(vesselScores) == 4
        vesselScores = ...
            vesselScores(:,:,:,1);
    end

    vesselProbSmall = ...
        vesselScores(:,:,2);

    vesselProb = imresize( ...
        vesselProbSmall, ...
        [h w], ...
        'bilinear');

    % Slightly relaxed threshold because DRIVE and IDRiD
    % are different datasets.
    vesselMask = vesselProb >= 0.35;

    vesselMask = ...
        vesselMask & retinaMask;

    vesselMask = ...
        bwareaopen(vesselMask,3);

    %% Keep vessel candidates away from HE annotation

    lesionSafety = imdilate( ...
        HE, ...
        strel('disk',12,0));

    vesselCandidates = ...
        vesselMask & ~lesionSafety;

    %% =====================================================
    % 1. HAEMORRHAGE POSITIVE PATCHES
    %% =====================================================

    [rr,cc] = find(HE);

    createdPos = 0;

    if ~isempty(rr)

        for p = 1:positivePerImage

            k = randi(numel(rr));

            centerR = rr(k) + randi([-48 48]);
            centerC = cc(k) + randi([-48 48]);

            r1 = round(centerR-patchSize/2);
            c1 = round(centerC-patchSize/2);

            r1 = max(1,min(r1,h-patchSize+1));
            c1 = max(1,min(c1,w-patchSize+1));

            rows = r1:r1+patchSize-1;
            cols = c1:c1+patchSize-1;

            Ipatch = I(rows,cols,:);
            Mpatch = HE(rows,cols);

            % Keep only genuinely positive patches
            if nnz(Mpatch) == 0
                continue
            end

            counter = counter + 1;
            positiveCount = positiveCount + 1;
            createdPos = createdPos + 1;

            filename = sprintf( ...
                'he_pos_%06d.png',counter);

            imwrite(Ipatch, ...
                fullfile(imageOut,filename));

            imwrite(uint8(Mpatch), ...
                fullfile(maskOut,filename));

        end
    end

    %% =====================================================
    % 2. VESSEL HARD NEGATIVES
    %
    % These patches contain vessels but ZERO haemorrhage
    %% =====================================================

    [vr,vc] = find(vesselCandidates);

    createdVesselNeg = 0;
    attempts = 0;

    while createdVesselNeg < vesselNegPerImage && ...
            attempts < 1000 && ...
            ~isempty(vr)

        attempts = attempts + 1;

        k = randi(numel(vr));

        centerR = vr(k);
        centerC = vc(k);

        r1 = round(centerR-patchSize/2);
        c1 = round(centerC-patchSize/2);

        r1 = max(1,min(r1,h-patchSize+1));
        c1 = max(1,min(c1,w-patchSize+1));

        rows = r1:r1+patchSize-1;
        cols = c1:c1+patchSize-1;

        HEPatch = HE(rows,cols);
        retinaPatch = retinaMask(rows,cols);
        vesselPatch = vesselMask(rows,cols);

        % Important conditions:
        % 1. no haemorrhage
        % 2. mostly inside retina
        % 3. contains meaningful vessel pixels

        if nnz(HEPatch) == 0 && ...
                mean(retinaPatch(:)) > 0.70 && ...
                nnz(vesselPatch) >= 20

            Ipatch = I(rows,cols,:);

            counter = counter + 1;
            vesselNegCount = vesselNegCount + 1;
            createdVesselNeg = createdVesselNeg + 1;

            filename = sprintf( ...
                'he_vesselneg_%06d.png',counter);

            imwrite(Ipatch, ...
                fullfile(imageOut,filename));

            imwrite(uint8(HEPatch), ...
                fullfile(maskOut,filename));

        end
    end

    %% =====================================================
    % 3. NORMAL RETINAL NEGATIVES
    %% =====================================================

    createdNormalNeg = 0;
    attempts = 0;

    while createdNormalNeg < normalNegPerImage && ...
            attempts < 1000

        attempts = attempts + 1;

        centerR = randi(h);
        centerC = randi(w);

        r1 = round(centerR-patchSize/2);
        c1 = round(centerC-patchSize/2);

        r1 = max(1,min(r1,h-patchSize+1));
        c1 = max(1,min(c1,w-patchSize+1));

        rows = r1:r1+patchSize-1;
        cols = c1:c1+patchSize-1;

        retinaPatch = retinaMask(rows,cols);
        HEPatch = HE(rows,cols);
        vesselPatch = vesselMask(rows,cols);

        % Normal negative:
        % no HE + mostly retina +
        % avoid extremely vessel-dense regions

        if nnz(HEPatch) == 0 && ...
                mean(retinaPatch(:)) > 0.75 && ...
                mean(vesselPatch(:)) < 0.10

            Ipatch = I(rows,cols,:);

            counter = counter + 1;
            normalNegCount = normalNegCount + 1;
            createdNormalNeg = createdNormalNeg + 1;

            filename = sprintf( ...
                'he_normalneg_%06d.png',counter);

            imwrite(Ipatch, ...
                fullfile(imageOut,filename));

            imwrite(uint8(HEPatch), ...
                fullfile(maskOut,filename));

        end
    end

    fprintf('   Positive      : %d\n',createdPos);
    fprintf('   Vessel neg    : %d\n',createdVesselNeg);
    fprintf('   Normal neg    : %d\n',createdNormalNeg);

end

%% =========================================================
% Final statistics
%% =========================================================

fprintf('\n========================================\n');
fprintf('HAEMORRHAGE V2 PATCH DATASET COMPLETE\n');
fprintf('========================================\n');

fprintf('Positive patches       : %d\n', ...
    positiveCount);

fprintf('Vessel hard negatives  : %d\n', ...
    vesselNegCount);

fprintf('Normal negatives       : %d\n', ...
    normalNegCount);

fprintf('----------------------------------------\n');

fprintf('TOTAL PATCHES           : %d\n', ...
    counter);

fprintf('========================================\n');

%% =========================================================
% Save V2 split information
%% =========================================================

save('models/haemorrhageV2Split.mat', ...
    'trainHEIdx', ...
    'valHEIdx', ...
    'positiveCount', ...
    'vesselNegCount', ...
    'normalNegCount');

fprintf('\nSaved: models/haemorrhageV2Split.mat\n');