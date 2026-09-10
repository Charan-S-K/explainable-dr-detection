%% SeeBeyond - Soft Exudate V2 Patch Generator
% Accuracy improvement experiment
%
% Creates:
%   1. Soft-exudate positive patches
%   2. Hard-exudate hard negatives
%   3. Optic-disc hard negatives
%   4. Normal retinal negatives
%
% Existing baseline model is NOT modified.

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

rng(42);

fprintf('\n========================================\n');
fprintf('SOFT EXUDATE V2 PATCH GENERATION\n');
fprintf('========================================\n');

%% =========================================================
% Load same split
%% =========================================================

load('models/softExudateSplit.mat', ...
    'trainSEIdx','valSEIdx');

fprintf('Training images   : %d\n',numel(trainSEIdx));
fprintf('Validation images : %d\n',numel(valSEIdx));

%% =========================================================
% Dataset paths
%% =========================================================

idridRoot = fullfile(pwd, ...
    'datasets','IDRiD','Segmentation','A. Segmentation');

imageRoot = fullfile(idridRoot, ...
    '1. Original Images','a. Training Set');

seMaskRoot = fullfile(pwd, ...
    'datasets','IDRiD','processed_masks','soft_exudates');

exMaskRoot = fullfile(pwd, ...
    'datasets','IDRiD','processed_masks','hard_exudates');

%% Original IDRiD optic-disc ground truth folder

odMaskRoot = fullfile(idridRoot, ...
    '2. All Segmentation Groundtruths', ...
    'a. Training Set', ...
    '5. Optic Disc');

if ~isfolder(odMaskRoot)
    error('Optic Disc ground-truth folder not found: %s',odMaskRoot);
end

%% New V2 patch directory

patchRoot = fullfile(pwd, ...
    'datasets','IDRiD','patches','soft_exudates_v2');

imageOut = fullfile(patchRoot,'images');
maskOut  = fullfile(patchRoot,'masks');

if exist(patchRoot,'dir')
    rmdir(patchRoot,'s');
end

mkdir(imageOut);
mkdir(maskOut);

%% =========================================================
% Datastore
%% =========================================================

imds = imageDatastore(imageRoot);

patchSize = 256;

positivePerImage      = 50;
hardExNegPerImage     = 20;
opticDiscNegPerImage  = 15;
normalNegPerImage     = 15;

counter = 0;

positiveCount = 0;
hardExNegCount = 0;
opticDiscNegCount = 0;
normalNegCount = 0;

positiveImageCount = 0;

%% =========================================================
% Process training images only
%% =========================================================

for n = 1:numel(trainSEIdx)

    idx = trainSEIdx(n);

    I = readimage(imds,idx);

    [~,imageID,~] = fileparts(imds.Files{idx});

    fprintf('\n[%d/%d] %s\n', ...
        n,numel(trainSEIdx),imageID);

    %% -----------------------------------------
    % Soft Exudate mask
    %% -----------------------------------------

    SE = imread( ...
        fullfile(seMaskRoot,[imageID '.png'])) > 0;

    %% -----------------------------------------
    % Hard Exudate mask
    %% -----------------------------------------

    EX = imread( ...
        fullfile(exMaskRoot,[imageID '.png'])) > 0;

    %% -----------------------------------------
    % Optic Disc mask
    %
    % Use wildcard so we don't depend on exact
    % IDRiD suffix/extension naming.
    %% -----------------------------------------

    odFiles = dir(fullfile(odMaskRoot, ...
        [imageID '*']));

    OD = false(size(SE));

    if ~isempty(odFiles)

        odPath = fullfile( ...
            odFiles(1).folder, ...
            odFiles(1).name);

        OD = imread(odPath) > 0;

        if ~isequal(size(OD),size(SE))
            OD = imresize(OD,size(SE),'nearest');
        end
    end

    %% -----------------------------------------
    % Retina mask
    %% -----------------------------------------

    [h,w,~] = size(I);

    G = im2double(rgb2gray(I));

    retinaMask = G > 0.03;

    if any(retinaMask(:))
        retinaMask = bwareafilt(retinaMask,1);
        retinaMask = imfill(retinaMask,'holes');
    end

    %% =====================================================
    % 1. SOFT-EXUDATE POSITIVE PATCHES
    %% =====================================================

    [rr,cc] = find(SE);

    createdPos = 0;

    if ~isempty(rr)

        positiveImageCount = positiveImageCount + 1;

        for p = 1:positivePerImage

            k = randi(numel(rr));

            centerR = rr(k) + randi([-64 64]);
            centerC = cc(k) + randi([-64 64]);

            r1 = round(centerR - patchSize/2);
            c1 = round(centerC - patchSize/2);

            r1 = max(1,min(r1,h-patchSize+1));
            c1 = max(1,min(c1,w-patchSize+1));

            rows = r1:r1+patchSize-1;
            cols = c1:c1+patchSize-1;

            Mpatch = SE(rows,cols);

            if nnz(Mpatch) == 0
                continue
            end

            Ipatch = I(rows,cols,:);

            counter = counter + 1;
            positiveCount = positiveCount + 1;
            createdPos = createdPos + 1;

            filename = sprintf( ...
                'se_pos_%06d.png',counter);

            imwrite(Ipatch, ...
                fullfile(imageOut,filename));

            imwrite(uint8(Mpatch), ...
                fullfile(maskOut,filename));
        end
    end

    %% =====================================================
    % 2. HARD-EXUDATE HARD NEGATIVES
    %% =====================================================

    hardRegion = EX & ~SE;

    [hr,hc] = find(hardRegion);

    createdHard = 0;
    attempts = 0;

    while createdHard < hardExNegPerImage && ...
            attempts < 1000 && ...
            ~isempty(hr)

        attempts = attempts + 1;

        k = randi(numel(hr));

        centerR = hr(k);
        centerC = hc(k);

        r1 = round(centerR-patchSize/2);
        c1 = round(centerC-patchSize/2);

        r1 = max(1,min(r1,h-patchSize+1));
        c1 = max(1,min(c1,w-patchSize+1));

        rows = r1:r1+patchSize-1;
        cols = c1:c1+patchSize-1;

        SEPatch = SE(rows,cols);
        retinaPatch = retinaMask(rows,cols);
        EXPatch = EX(rows,cols);

        if nnz(SEPatch) == 0 && ...
                mean(retinaPatch(:)) > 0.70 && ...
                nnz(EXPatch) >= 10

            Ipatch = I(rows,cols,:);

            counter = counter + 1;
            hardExNegCount = hardExNegCount + 1;
            createdHard = createdHard + 1;

            filename = sprintf( ...
                'se_hardexneg_%06d.png',counter);

            imwrite(Ipatch, ...
                fullfile(imageOut,filename));

            imwrite(uint8(SEPatch), ...
                fullfile(maskOut,filename));
        end
    end

    %% =====================================================
    % 3. OPTIC-DISC HARD NEGATIVES
    %% =====================================================

    [orr,occ] = find(OD);

    createdOD = 0;
    attempts = 0;

    while createdOD < opticDiscNegPerImage && ...
            attempts < 1000 && ...
            ~isempty(orr)

        attempts = attempts + 1;

        k = randi(numel(orr));

        centerR = orr(k) + randi([-40 40]);
        centerC = occ(k) + randi([-40 40]);

        r1 = round(centerR-patchSize/2);
        c1 = round(centerC-patchSize/2);

        r1 = max(1,min(r1,h-patchSize+1));
        c1 = max(1,min(c1,w-patchSize+1));

        rows = r1:r1+patchSize-1;
        cols = c1:c1+patchSize-1;

        SEPatch = SE(rows,cols);
        ODPatch = OD(rows,cols);
        retinaPatch = retinaMask(rows,cols);

        if nnz(SEPatch) == 0 && ...
                nnz(ODPatch) >= 20 && ...
                mean(retinaPatch(:)) > 0.60

            Ipatch = I(rows,cols,:);

            counter = counter + 1;
            opticDiscNegCount = opticDiscNegCount + 1;
            createdOD = createdOD + 1;

            filename = sprintf( ...
                'se_odneg_%06d.png',counter);

            imwrite(Ipatch, ...
                fullfile(imageOut,filename));

            imwrite(uint8(SEPatch), ...
                fullfile(maskOut,filename));
        end
    end

    %% =====================================================
    % 4. NORMAL RETINAL NEGATIVES
    %% =====================================================

    createdNormal = 0;
    attempts = 0;

    while createdNormal < normalNegPerImage && ...
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

        SEPatch = SE(rows,cols);
        EXPatch = EX(rows,cols);
        ODPatch = OD(rows,cols);
        retinaPatch = retinaMask(rows,cols);

        if nnz(SEPatch) == 0 && ...
                nnz(EXPatch) == 0 && ...
                nnz(ODPatch) == 0 && ...
                mean(retinaPatch(:)) > 0.75

            Ipatch = I(rows,cols,:);

            counter = counter + 1;
            normalNegCount = normalNegCount + 1;
            createdNormal = createdNormal + 1;

            filename = sprintf( ...
                'se_normalneg_%06d.png',counter);

            imwrite(Ipatch, ...
                fullfile(imageOut,filename));

            imwrite(uint8(SEPatch), ...
                fullfile(maskOut,filename));
        end
    end

    fprintf('   Positive       : %d\n',createdPos);
    fprintf('   HardEx negative: %d\n',createdHard);
    fprintf('   OpticDisc neg  : %d\n',createdOD);
    fprintf('   Normal negative: %d\n',createdNormal);

end

%% =========================================================
% Final report
%% =========================================================

fprintf('\n========================================\n');
fprintf('SOFT EXUDATE V2 PATCH DATASET COMPLETE\n');
fprintf('========================================\n');

fprintf('Positive training images : %d\n', ...
    positiveImageCount);

fprintf('Positive patches         : %d\n', ...
    positiveCount);

fprintf('Hard Exudate negatives   : %d\n', ...
    hardExNegCount);

fprintf('Optic Disc negatives     : %d\n', ...
    opticDiscNegCount);

fprintf('Normal negatives         : %d\n', ...
    normalNegCount);

fprintf('----------------------------------------\n');

fprintf('TOTAL PATCHES            : %d\n', ...
    counter);

fprintf('========================================\n');

%% Save split/statistics

save('models/softExudateV2Split.mat', ...
    'trainSEIdx', ...
    'valSEIdx', ...
    'positiveImageCount', ...
    'positiveCount', ...
    'hardExNegCount', ...
    'opticDiscNegCount', ...
    'normalNegCount');

fprintf('\nSaved: models/softExudateV2Split.mat\n');
