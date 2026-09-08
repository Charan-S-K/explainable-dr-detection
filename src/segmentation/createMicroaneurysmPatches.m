%% SeeBeyond - Microaneurysm Patch Generator
% Original-resolution lesion-focused training

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

rng(42);

fprintf('\n===== MICROANEURYSM PATCH GENERATION =====\n');

%% ------------------------------------------------
% Use the same train/validation split
%% ------------------------------------------------

load('models/hardExudateNetV2.mat', ...
    'trainEXIdx','valEXIdx');

trainMAIdx = trainEXIdx;
valMAIdx   = valEXIdx;

%% ------------------------------------------------
% Paths
%% ------------------------------------------------

idridRoot = fullfile(pwd, ...
    'datasets','IDRiD','Segmentation','A. Segmentation');

imageRoot = fullfile(idridRoot, ...
    '1. Original Images','a. Training Set');

maMaskRoot = fullfile(pwd, ...
    'datasets','IDRiD','processed_masks','microaneurysms');

heMaskRoot = fullfile(pwd, ...
    'datasets','IDRiD','processed_masks','haemorrhages');

patchRoot = fullfile(pwd, ...
    'datasets','IDRiD','patches','microaneurysms');

imageOut = fullfile(patchRoot,'images');
maskOut  = fullfile(patchRoot,'masks');

%% Clean previous MA patches only

if exist(patchRoot,'dir')
    rmdir(patchRoot,'s');
end

mkdir(imageOut);
mkdir(maskOut);

%% ------------------------------------------------
% Datastore
%% ------------------------------------------------

imds = imageDatastore(imageRoot);

patchSize = 256;

positivePerImage     = 40;
heHardNegPerImage    = 15;
randomNegPerImage    = 15;

counter = 0;

%% ------------------------------------------------
% Process ONLY training images
%% ------------------------------------------------

for n = 1:numel(trainMAIdx)

    idx = trainMAIdx(n);

    I = readimage(imds,idx);

    [~,imageID,~] = fileparts(imds.Files{idx});

    maPath = fullfile(maMaskRoot,[imageID '.png']);
    hePath = fullfile(heMaskRoot,[imageID '.png']);

    MA = imread(maPath) > 0;

    if isfile(hePath)
        HE = imread(hePath) > 0;
    else
        HE = false(size(MA));
    end

    [h,w,~] = size(I);

    %% Retina field

    G = im2double(rgb2gray(I));

    retinaMask = G > 0.03;
    retinaMask = bwareafilt(retinaMask,1);
    retinaMask = imfill(retinaMask,'holes');

    %% ============================================
    % 1. MICROANEURYSM POSITIVE PATCHES
    %% ============================================

    [rr,cc] = find(MA);

    if ~isempty(rr)

        for p = 1:positivePerImage

            k = randi(numel(rr));

            centerR = rr(k) + randi([-48 48]);
            centerC = cc(k) + randi([-48 48]);

            r1 = round(centerR - patchSize/2);
            c1 = round(centerC - patchSize/2);

            r1 = max(1,min(r1,h-patchSize+1));
            c1 = max(1,min(c1,w-patchSize+1));

            rows = r1:r1+patchSize-1;
            cols = c1:c1+patchSize-1;

            Ipatch = I(rows,cols,:);
            Mpatch = MA(rows,cols);

            counter = counter + 1;

            filename = sprintf( ...
                'ma_pos_%06d.png',counter);

            imwrite(Ipatch, ...
                fullfile(imageOut,filename));

            imwrite(uint8(Mpatch), ...
                fullfile(maskOut,filename));
        end
    end

    %% ============================================
    % 2. HAEMORRHAGE HARD-NEGATIVE PATCHES
    %
    % Helps MA model distinguish haemorrhage
    % from tiny microaneurysms.
    %% ============================================

    [hr,hc] = find(HE);

    createdHard = 0;
    attemptsHard = 0;

    while createdHard < heHardNegPerImage && ...
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

        MAPatch = MA(rows,cols);
        retinaPatch = retinaMask(rows,cols);

        % Prefer haemorrhage regions without MA
        if nnz(MAPatch) == 0 && ...
                mean(retinaPatch(:)) > 0.60

            Ipatch = I(rows,cols,:);

            counter = counter + 1;

            filename = sprintf( ...
                'ma_hardneg_%06d.png',counter);

            imwrite(Ipatch, ...
                fullfile(imageOut,filename));

            imwrite(uint8(MAPatch), ...
                fullfile(maskOut,filename));

            createdHard = createdHard + 1;
        end
    end

    %% ============================================
    % 3. RANDOM NORMAL RETINA NEGATIVES
    %% ============================================

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
        MAPatch = MA(rows,cols);

        if mean(retinaPatch(:)) > 0.70 && ...
                nnz(MAPatch) == 0

            Ipatch = I(rows,cols,:);

            counter = counter + 1;

            filename = sprintf( ...
                'ma_neg_%06d.png',counter);

            imwrite(Ipatch, ...
                fullfile(imageOut,filename));

            imwrite(uint8(MAPatch), ...
                fullfile(maskOut,filename));

            createdNeg = createdNeg + 1;
        end
    end

    fprintf('Processed MA image %d / %d\n', ...
        n,numel(trainMAIdx));

end

%% ------------------------------------------------
% Finish
%% ------------------------------------------------

fprintf('\n========================================\n');
fprintf('Microaneurysm patches created: %d\n',counter);
fprintf('Images: %s\n',imageOut);
fprintf('Masks : %s\n',maskOut);
fprintf('========================================\n');

save('models/microaneurysmSplit.mat', ...
    'trainMAIdx','valMAIdx');