%% SeeBeyond - Create Hard Exudate Training Patches V5
%
% Improved patch sampling:
%   1. Lesion-centered patches
%   2. Dense-lesion patches
%   3. Lesion-boundary patches
%   4. Optic-disc hard negatives
%   5. Vessel-region hard negatives
%   6. Bright retinal hard negatives
%   7. Normal retinal negatives
%
% Uses the SAME 43/11 train-validation split as Hard Exudate V2.

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

rng(42);

fprintf('\n=============================================\n');
fprintf(' SeeBeyond - Hard Exudate Patch Dataset V5\n');
fprintf('=============================================\n');

%% Load SAME split as V2

S = load('models/hardExudateNetV2.mat', ...
    'trainEXIdx','valEXIdx');

trainEXIdx = S.trainEXIdx;
valEXIdx   = S.valEXIdx;

fprintf('Training images   : %d\n',numel(trainEXIdx));
fprintf('Validation images : %d\n',numel(valEXIdx));

%% Paths

idridRoot = fullfile(pwd, ...
    'datasets','IDRiD','Segmentation','A. Segmentation');

imageRoot = fullfile(idridRoot, ...
    '1. Original Images','a. Training Set');

maskRoot = fullfile(pwd, ...
    'datasets','IDRiD','processed_masks','hard_exudates');

patchRoot = fullfile(pwd, ...
    'datasets','IDRiD','patches','hard_exudatesV5');

patchImageDir = fullfile(patchRoot,'images');
patchMaskDir  = fullfile(patchRoot,'masks');

%% Recreate directories

if exist(patchImageDir,'dir')
    rmdir(patchImageDir,'s');
end

if exist(patchMaskDir,'dir')
    rmdir(patchMaskDir,'s');
end

mkdir(patchImageDir);
mkdir(patchMaskDir);

%% Datastore

imds = imageDatastore(imageRoot);

patchSize = 256;

%% Patch counts per image

positivePatchesPerImage = 35;
densePatchesPerImage    = 15;
boundaryPatchesPerImage = 10;

opticDiscNegativePerImage = 15;
vesselNegativePerImage    = 15;
brightNegativePerImage    = 10;
normalNegativePerImage    = 10;

patchCounter = 0;

fprintf('\nPatch configuration:\n');
fprintf('  Lesion-centered : %d\n',positivePatchesPerImage);
fprintf('  Dense lesion    : %d\n',densePatchesPerImage);
fprintf('  Boundary        : %d\n',boundaryPatchesPerImage);
fprintf('  Optic disc      : %d\n',opticDiscNegativePerImage);
fprintf('  Vessel          : %d\n',vesselNegativePerImage);
fprintf('  Bright retinal  : %d\n',brightNegativePerImage);
fprintf('  Normal retinal  : %d\n',normalNegativePerImage);

%% Helper for saving patches

savePatch = @(Ipatch,Mpatch) saveOnePatch( ...
    Ipatch,Mpatch,patchImageDir,patchMaskDir,patchCounter);

%% ============================================================
% PROCESS TRAINING IMAGES
% =============================================================

for n = 1:numel(trainEXIdx)

    i = trainEXIdx(n);

    imagePath = imds.Files{i};

    [~,imageID,~] = fileparts(imagePath);

    maskPath = fullfile(maskRoot,imageID + ".png");

    I = imread(imagePath);
    M = imread(maskPath);

    if size(I,3) == 1
        I = repmat(I,[1 1 3]);
    end

    if size(M,3) > 1
        M = M(:,:,1);
    end

    M = M > 0;

    [h,w,~] = size(I);

    %% ---------------------------------------------------------
    % Retina mask
    % ----------------------------------------------------------

    G = im2double(rgb2gray(I));

    retinaMask = G > 0.03;

    retinaMask = bwareafilt(retinaMask,1);

    retinaMask = imfill(retinaMask,'holes');

    %% ---------------------------------------------------------
    % Vessel approximation
    %
    % Dark thin structures are useful hard negatives.
    % ----------------------------------------------------------

    vesselApprox = G < 0.18;

    vesselApprox = vesselApprox & retinaMask;

    vesselApprox = bwareaopen(vesselApprox,20);

    vesselApprox = imdilate( ...
        vesselApprox, ...
        strel('disk',2));

    %% ---------------------------------------------------------
    % Bright retinal regions
    %
    % Exudates are bright, so bright non-lesion regions
    % make useful hard negatives.
    % ----------------------------------------------------------

    brightMask = G > 0.72;

    brightMask = brightMask & retinaMask;

    brightMask = brightMask & ~M;

    brightMask = bwareaopen(brightMask,30);

    %% ---------------------------------------------------------
    % Approximate optic disc
    %
    % The optic disc is generally one of the brightest
    % non-lesion structures.
    % ----------------------------------------------------------

    discCandidate = G > 0.80;

    discCandidate = discCandidate & retinaMask;

    discCandidate = bwareafilt(discCandidate,1);

    discCandidate = imfill(discCandidate,'holes');

    discCandidate = imdilate( ...
        discCandidate, ...
        strel('disk',20));

    discCandidate = discCandidate & ~M;

    %% =========================================================
    % 1. LESION-CENTERED PATCHES
    % =========================================================

    [lesionRows,lesionCols] = find(M);

    if ~isempty(lesionRows)

        for p = 1:positivePatchesPerImage

            pick = randi(numel(lesionRows));

            centerR = lesionRows(pick);
            centerC = lesionCols(pick);

            % Smaller shift preserves lesion content.
            centerR = centerR + randi([-32 32]);
            centerC = centerC + randi([-32 32]);

            r1 = round(centerR - patchSize/2);
            c1 = round(centerC - patchSize/2);

            r1 = max(1,min(r1,h-patchSize+1));
            c1 = max(1,min(c1,w-patchSize+1));

            rows = r1:(r1+patchSize-1);
            cols = c1:(c1+patchSize-1);

            Ipatch = I(rows,cols,:);
            Mpatch = M(rows,cols);

            patchCounter = patchCounter + 1;

            filename = sprintf('patch_%06d.png',patchCounter);

            imwrite(Ipatch, ...
                fullfile(patchImageDir,filename));

            imwrite(uint8(Mpatch), ...
                fullfile(patchMaskDir,filename));
        end
    end

    %% =========================================================
    % 2. DENSE LESION PATCHES
    % =========================================================

    if ~isempty(lesionRows)

        lesionPixelCount = numel(lesionRows);

        for p = 1:densePatchesPerImage

            bestFound = false;

            for attempt = 1:100

                pick = randi(lesionPixelCount);

                centerR = lesionRows(pick);
                centerC = lesionCols(pick);

                r1 = round(centerR - patchSize/2);
                c1 = round(centerC - patchSize/2);

                r1 = max(1,min(r1,h-patchSize+1));
                c1 = max(1,min(c1,w-patchSize+1));

                rows = r1:(r1+patchSize-1);
                cols = c1:(c1+patchSize-1);

                Mpatch = M(rows,cols);

                lesionFraction = mean(Mpatch(:));

                if lesionFraction > 0.015

                    Ipatch = I(rows,cols,:);

                    patchCounter = patchCounter + 1;

                    filename = sprintf( ...
                        'patch_%06d.png',patchCounter);

                    imwrite(Ipatch, ...
                        fullfile(patchImageDir,filename));

                    imwrite(uint8(Mpatch), ...
                        fullfile(patchMaskDir,filename));

                    bestFound = true;
                    break;
                end
            end

            if ~bestFound
                % Skip if no dense patch was found.
            end
        end
    end

    %% =========================================================
    % 3. LESION BOUNDARY PATCHES
    % =========================================================

    if ~isempty(lesionRows)

        boundaryMask = bwperim(M);

        [boundaryRows,boundaryCols] = find(boundaryMask);

        if ~isempty(boundaryRows)

            for p = 1:boundaryPatchesPerImage

                pick = randi(numel(boundaryRows));

                centerR = boundaryRows(pick);
                centerC = boundaryCols(pick);

                r1 = round(centerR - patchSize/2);
                c1 = round(centerC - patchSize/2);

                r1 = max(1,min(r1,h-patchSize+1));
                c1 = max(1,min(c1,w-patchSize+1));

                rows = r1:(r1+patchSize-1);
                cols = c1:(c1+patchSize-1);

                Ipatch = I(rows,cols,:);
                Mpatch = M(rows,cols);

                patchCounter = patchCounter + 1;

                filename = sprintf( ...
                    'patch_%06d.png',patchCounter);

                imwrite(Ipatch, ...
                    fullfile(patchImageDir,filename));

                imwrite(uint8(Mpatch), ...
                    fullfile(patchMaskDir,filename));
            end
        end
    end

    %% =========================================================
    % 4. OPTIC DISC HARD NEGATIVES
    % =========================================================

    negativeCreated = 0;
    attempts = 0;

    [discRows,discCols] = find(discCandidate);

    while negativeCreated < opticDiscNegativePerImage && ...
            attempts < 500

        attempts = attempts + 1;

        if isempty(discRows)
            break;
        end

        pick = randi(numel(discRows));

        centerR = discRows(pick);
        centerC = discCols(pick);

        r1 = round(centerR - patchSize/2);
        c1 = round(centerC - patchSize/2);

        r1 = max(1,min(r1,h-patchSize+1));
        c1 = max(1,min(c1,w-patchSize+1));

        rows = r1:(r1+patchSize-1);
        cols = c1:(c1+patchSize-1);

        Mpatch = M(rows,cols);

        if nnz(Mpatch) == 0

            Ipatch = I(rows,cols,:);

            patchCounter = patchCounter + 1;

            filename = sprintf( ...
                'patch_%06d.png',patchCounter);

            imwrite(Ipatch, ...
                fullfile(patchImageDir,filename));

            imwrite(uint8(Mpatch), ...
                fullfile(patchMaskDir,filename));

            negativeCreated = negativeCreated + 1;
        end
    end

    %% =========================================================
    % 5. VESSEL HARD NEGATIVES
    % =========================================================

    negativeCreated = 0;
    attempts = 0;

    [vesselRows,vesselCols] = find(vesselApprox);

    while negativeCreated < vesselNegativePerImage && ...
            attempts < 1000

        attempts = attempts + 1;

        if isempty(vesselRows)
            break;
        end

        pick = randi(numel(vesselRows));

        centerR = vesselRows(pick);
        centerC = vesselCols(pick);

        r1 = round(centerR - patchSize/2);
        c1 = round(centerC - patchSize/2);

        r1 = max(1,min(r1,h-patchSize+1));
        c1 = max(1,min(c1,w-patchSize+1));

        rows = r1:(r1+patchSize-1);
        cols = c1:(c1+patchSize-1);

        Mpatch = M(rows,cols);

        if nnz(Mpatch) == 0

            Ipatch = I(rows,cols,:);

            patchCounter = patchCounter + 1;

            filename = sprintf( ...
                'patch_%06d.png',patchCounter);

            imwrite(Ipatch, ...
                fullfile(patchImageDir,filename));

            imwrite(uint8(Mpatch), ...
                fullfile(patchMaskDir,filename));

            negativeCreated = negativeCreated + 1;
        end
    end

    %% =========================================================
    % 6. BRIGHT NON-EXUDATE HARD NEGATIVES
    % =========================================================

    negativeCreated = 0;
    attempts = 0;

    [brightRows,brightCols] = find(brightMask);

    while negativeCreated < brightNegativePerImage && ...
            attempts < 1000

        attempts = attempts + 1;

        if isempty(brightRows)
            break;
        end

        pick = randi(numel(brightRows));

        centerR = brightRows(pick);
        centerC = brightCols(pick);

        r1 = round(centerR - patchSize/2);
        c1 = round(centerC - patchSize/2);

        r1 = max(1,min(r1,h-patchSize+1));
        c1 = max(1,min(c1,w-patchSize+1));

        rows = r1:(r1+patchSize-1);
        cols = c1:(c1+patchSize-1);

        Mpatch = M(rows,cols);

        if nnz(Mpatch) == 0

            Ipatch = I(rows,cols,:);

            patchCounter = patchCounter + 1;

            filename = sprintf( ...
                'patch_%06d.png',patchCounter);

            imwrite(Ipatch, ...
                fullfile(patchImageDir,filename));

            imwrite(uint8(Mpatch), ...
                fullfile(patchMaskDir,filename));

            negativeCreated = negativeCreated + 1;
        end
    end

    %% =========================================================
    % 7. NORMAL RETINAL NEGATIVES
    % =========================================================

    negativeCreated = 0;
    attempts = 0;

    while negativeCreated < normalNegativePerImage && ...
            attempts < 1000

        attempts = attempts + 1;

        centerR = randi(h);
        centerC = randi(w);

        r1 = round(centerR - patchSize/2);
        c1 = round(centerC - patchSize/2);

        r1 = max(1,min(r1,h-patchSize+1));
        c1 = max(1,min(c1,w-patchSize+1));

        rows = r1:(r1+patchSize-1);
        cols = c1:(c1+patchSize-1);

        retinaPatch = retinaMask(rows,cols);
        Mpatch = M(rows,cols);

        if mean(retinaPatch(:)) > 0.70 && ...
                nnz(Mpatch) == 0

            Ipatch = I(rows,cols,:);

            patchCounter = patchCounter + 1;

            filename = sprintf( ...
                'patch_%06d.png',patchCounter);

            imwrite(Ipatch, ...
                fullfile(patchImageDir,filename));

            imwrite(uint8(Mpatch), ...
                fullfile(patchMaskDir,filename));

            negativeCreated = negativeCreated + 1;
        end
    end

    fprintf('Processed image %02d / %02d: %s\n', ...
        n,numel(trainEXIdx),imageID);

end

%% ============================================================
% SUMMARY
% =============================================================

fprintf('\n=============================================\n');
fprintf(' HARD EXUDATE V5 PATCH CREATION COMPLETE\n');
fprintf('=============================================\n');

fprintf('Total patches: %d\n',patchCounter);
fprintf('Images       : %s\n',patchImageDir);
fprintf('Masks        : %s\n',patchMaskDir);

fprintf('=============================================\n');


%% Local helper function
function saveOnePatch(Ipatch,Mpatch,imageDir,maskDir,counter)

filename = sprintf('patch_%06d.png',counter);

imwrite(Ipatch, ...
    fullfile(imageDir,filename));

imwrite(uint8(Mpatch), ...
    fullfile(maskDir,filename));

end