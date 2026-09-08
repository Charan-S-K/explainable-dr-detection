%% SeeBeyond - Soft Exudate Patch Generator

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

rng(42);

fprintf('\n===== SOFT EXUDATE PATCH GENERATION =====\n');

%% Same train/validation split

load('models/hardExudateNetV2.mat', ...
    'trainEXIdx','valEXIdx');

trainSEIdx = trainEXIdx;
valSEIdx   = valEXIdx;

%% Paths

idridRoot = fullfile(pwd, ...
    'datasets','IDRiD','Segmentation','A. Segmentation');

imageRoot = fullfile(idridRoot, ...
    '1. Original Images','a. Training Set');

seMaskRoot = fullfile(pwd, ...
    'datasets','IDRiD','processed_masks','soft_exudates');

exMaskRoot = fullfile(pwd, ...
    'datasets','IDRiD','processed_masks','hard_exudates');

patchRoot = fullfile(pwd, ...
    'datasets','IDRiD','patches','soft_exudates');

imageOut = fullfile(patchRoot,'images');
maskOut  = fullfile(patchRoot,'masks');

if exist(patchRoot,'dir')
    rmdir(patchRoot,'s');
end

mkdir(imageOut);
mkdir(maskOut);

imds = imageDatastore(imageRoot);

patchSize = 256;

positivePerImage  = 40;
hardNegPerImage   = 20;
randomNegPerImage = 15;

counter = 0;

positiveTrainingImages = 0;

%% Generate patches

for n = 1:numel(trainSEIdx)

    idx = trainSEIdx(n);

    I = readimage(imds,idx);

    [~,imageID,~] = fileparts(imds.Files{idx});

    sePath = fullfile(seMaskRoot,[imageID '.png']);
    exPath = fullfile(exMaskRoot,[imageID '.png']);

    SE = imread(sePath) > 0;

    if isfile(exPath)
        EX = imread(exPath) > 0;
    else
        EX = false(size(SE));
    end

    [h,w,~] = size(I);

    %% Retina mask

    G = im2double(rgb2gray(I));

    retinaMask = G > 0.03;
    retinaMask = bwareafilt(retinaMask,1);
    retinaMask = imfill(retinaMask,'holes');

    %% ======================================
    % 1. Positive Soft Exudate patches
    %% ======================================

    [rr,cc] = find(SE);

    if ~isempty(rr)

        positiveTrainingImages = positiveTrainingImages + 1;

        for p = 1:positivePerImage

            k = randi(numel(rr));

            centerR = rr(k) + randi([-64 64]);
            centerC = cc(k) + randi([-64 64]);

            r1 = round(centerR-patchSize/2);
            c1 = round(centerC-patchSize/2);

            r1 = max(1,min(r1,h-patchSize+1));
            c1 = max(1,min(c1,w-patchSize+1));

            rows = r1:r1+patchSize-1;
            cols = c1:c1+patchSize-1;

            Ipatch = I(rows,cols,:);
            Mpatch = SE(rows,cols);

            counter = counter + 1;

            filename = sprintf( ...
                'se_pos_%06d.png',counter);

            imwrite(Ipatch, ...
                fullfile(imageOut,filename));

            imwrite(uint8(Mpatch), ...
                fullfile(maskOut,filename));

        end
    end

    %% ======================================
    % 2. Hard Exudate hard-negative patches
    %% ======================================

    hardRegion = EX & ~SE;

    [hr,hc] = find(hardRegion);

    createdHard = 0;
    attemptsHard = 0;

    while createdHard < hardNegPerImage && ...
            attemptsHard < 500 && ~isempty(hr)

        attemptsHard = attemptsHard + 1;

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

        % We want hard-exudate regions WITHOUT soft exudate
        if nnz(SEPatch) == 0 && ...
                mean(retinaPatch(:)) > 0.60

            Ipatch = I(rows,cols,:);

            counter = counter + 1;

            filename = sprintf( ...
                'se_hardneg_%06d.png',counter);

            imwrite(Ipatch, ...
                fullfile(imageOut,filename));

            imwrite(uint8(SEPatch), ...
                fullfile(maskOut,filename));

            createdHard = createdHard + 1;
        end
    end

    %% ======================================
    % 3. Random normal retinal patches
    %% ======================================

    createdNeg = 0;
    attemptsNeg = 0;

    while createdNeg < randomNegPerImage && ...
            attemptsNeg < 500

        attemptsNeg = attemptsNeg + 1;

        centerR = randi(h);
        centerC = randi(w);

        r1 = round(centerR-patchSize/2);
        c1 = round(centerC-patchSize/2);

        r1 = max(1,min(r1,h-patchSize+1));
        c1 = max(1,min(c1,w-patchSize+1));

        rows = r1:r1+patchSize-1;
        cols = c1:c1+patchSize-1;

        retinaPatch = retinaMask(rows,cols);
        SEPatch = SE(rows,cols);

        if mean(retinaPatch(:)) > 0.70 && ...
                nnz(SEPatch) == 0

            Ipatch = I(rows,cols,:);

            counter = counter + 1;

            filename = sprintf( ...
                'se_neg_%06d.png',counter);

            imwrite(Ipatch, ...
                fullfile(imageOut,filename));

            imwrite(uint8(SEPatch), ...
                fullfile(maskOut,filename));

            createdNeg = createdNeg + 1;
        end
    end

    fprintf('Processed SE image %d / %d\n', ...
        n,numel(trainSEIdx));

end

%% Save split

save('models/softExudateSplit.mat', ...
    'trainSEIdx','valSEIdx');

fprintf('\n========================================\n');
fprintf('Soft Exudate patches created : %d\n',counter);
fprintf('Positive training images     : %d\n', ...
    positiveTrainingImages);
fprintf('========================================\n');