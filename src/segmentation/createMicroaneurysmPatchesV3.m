%% SeeBeyond - Microaneurysm V3 Patch Generator
%
% V3 changes compared with V2:
%   - More MA-positive patches
%   - Fewer vessel hard negatives
%   - Fewer random negatives
%   - Keeps haemorrhage hard negatives
%   - Uses the EXISTING trained vessel model only to locate vessel regions
%   - Does NOT use DRIVE vessel masks directly on IDRiD images
%
% V1 is preserved.
% V2 is preserved.
%
% Patch categories:
%   1. Microaneurysm positive
%   2. Haemorrhage hard negative
%   3. Vessel hard negative
%   4. Random normal-retina negative

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

rng(42);

fprintf('\n');
fprintf('====================================================\n');
fprintf('MICROANEURYSM V3 PATCH GENERATION\n');
fprintf('====================================================\n');

%% ---------------------------------------------------
% Load the SAME train/validation split used previously
% ----------------------------------------------------

load('models/hardExudateNetV2.mat', ...
    'trainEXIdx', ...
    'valEXIdx');

trainMAIdx = trainEXIdx;
valMAIdx   = valEXIdx;

fprintf('Training images   : %d\n',numel(trainMAIdx));
fprintf('Validation images : %d\n',numel(valMAIdx));

%% ---------------------------------------------------
% Paths
% ----------------------------------------------------

idridRoot = fullfile( ...
    pwd, ...
    'datasets','IDRiD','Segmentation','A. Segmentation');

imageRoot = fullfile( ...
    idridRoot, ...
    '1. Original Images','a. Training Set');

maMaskRoot = fullfile( ...
    pwd, ...
    'datasets','IDRiD','processed_masks','microaneurysms');

heMaskRoot = fullfile( ...
    pwd, ...
    'datasets','IDRiD','processed_masks','haemorrhages');

patchRoot = fullfile( ...
    pwd, ...
    'datasets','IDRiD','patches','microaneurysmsV3');

imageOut = fullfile(patchRoot,'images');
maskOut  = fullfile(patchRoot,'masks');

%% ---------------------------------------------------
% Recreate V3 directory
% ----------------------------------------------------

if exist(patchRoot,'dir')
    fprintf('\nRemoving previous V3 patch directory...\n');
    rmdir(patchRoot,'s');
end

mkdir(imageOut);
mkdir(maskOut);

%% ---------------------------------------------------
% Load existing vessel model
% ----------------------------------------------------

fprintf('\nLoading trained vessel model...\n');

S = load( ...
    'models/vesselNetWeighted.mat', ...
    'vesselNetWeighted');

vesselNetWeighted = S.vesselNetWeighted;

fprintf('Vessel model loaded.\n');

%% ---------------------------------------------------
% V3 patch strategy
% ----------------------------------------------------

patchSize = 256;

% IMPORTANT:
% Increase positive examples because V2 became too conservative.
positivePerImage = 55;

% Keep haemorrhage negatives, but don't overwhelm positives.
heHardNegPerImage = 10;

% Reduce vessel negatives substantially compared with V2.
vesselHardNegPerImage = 10;

% Reduce random negatives.
randomNegPerImage = 10;

fprintf('\nV3 patch strategy:\n');
fprintf('Positive MA patches       : %d / image\n', ...
    positivePerImage);

fprintf('Haemorrhage hard negatives: %d / image\n', ...
    heHardNegPerImage);

fprintf('Vessel hard negatives     : %d / image\n', ...
    vesselHardNegPerImage);

fprintf('Random negatives          : %d / image\n', ...
    randomNegPerImage);

%% ---------------------------------------------------
% Dataset
% ----------------------------------------------------

imds = imageDatastore(imageRoot);

counter = 0;

totalPositive = 0;
totalHEHard   = 0;
totalVessel   = 0;
totalRandom   = 0;

%% ---------------------------------------------------
% Process training images
% ----------------------------------------------------

for n = 1:numel(trainMAIdx)

    idx = trainMAIdx(n);

    I = readimage(imds,idx);

    [~,imageID,~] = fileparts(imds.Files{idx});

    %% Load MA mask
    maPath = fullfile( ...
        maMaskRoot, ...
        [imageID '.png']);

    if ~isfile(maPath)
        error( ...
            'Microaneurysm mask not found: %s', ...
            maPath);
    end

    MA = imread(maPath) > 0;

    %% Load haemorrhage mask
    hePath = fullfile( ...
        heMaskRoot, ...
        [imageID '.png']);

    if isfile(hePath)
        HE = imread(hePath) > 0;
    else
        HE = false(size(MA));
    end

    %% Image size
    [h,w,~] = size(I);

    %% ------------------------------------------------
    % Retina mask
    % -------------------------------------------------

    G = im2double(rgb2gray(I));

    retinaMask = G > 0.03;

    if any(retinaMask(:))

        retinaMask = bwareafilt( ...
            retinaMask,1);

        retinaMask = imfill( ...
            retinaMask,'holes');

    end

    %% ------------------------------------------------
    % Vessel prediction on IDRiD image
    % -------------------------------------------------

    vesselProb = predictResizedVesselMap( ...
        I, ...
        vesselNetWeighted, ...
        [256 256]);

    vesselMask = vesselProb >= 0.50;

    vesselMask = vesselMask & retinaMask;

    vesselMask = bwareaopen( ...
        vesselMask,2);

    %% ------------------------------------------------
    % 1. MICROANEURYSM POSITIVE PATCHES
    % -------------------------------------------------

    [rr,cc] = find(MA);

    createdPositive = 0;

    if ~isempty(rr)

        for p = 1:positivePerImage

            k = randi(numel(rr));

            centerR = rr(k) + randi([-40 40]);
            centerC = cc(k) + randi([-40 40]);

            [rows,cols] = getPatchCoordinates( ...
                centerR, ...
                centerC, ...
                patchSize, ...
                h, ...
                w);

            Ipatch = I(rows,cols,:);
            Mpatch = MA(rows,cols);

            counter = counter + 1;

            filename = sprintf( ...
                'ma_pos_%06d.png', ...
                counter);

            imwrite( ...
                Ipatch, ...
                fullfile(imageOut,filename));

            imwrite( ...
                uint8(Mpatch), ...
                fullfile(maskOut,filename));

            createdPositive = createdPositive + 1;

            totalPositive = totalPositive + 1;
        end
    end

    %% ------------------------------------------------
    % 2. HAEMORRHAGE HARD NEGATIVES
    % -------------------------------------------------

    [hr,hc] = find(HE);

    createdHE = 0;
    attemptsHE = 0;

    while ...
            createdHE < heHardNegPerImage && ...
            attemptsHE < 1000 && ...
            ~isempty(hr)

        attemptsHE = attemptsHE + 1;

        k = randi(numel(hr));

        centerR = hr(k);
        centerC = hc(k);

        [rows,cols] = getPatchCoordinates( ...
            centerR, ...
            centerC, ...
            patchSize, ...
            h, ...
            w);

        MAPatch = MA(rows,cols);
        retinaPatch = retinaMask(rows,cols);

        retinaFraction = mean(retinaPatch(:));

        % Must contain haemorrhage-like structure
        % but NO annotated microaneurysm.
        if nnz(MAPatch) == 0 && ...
                retinaFraction > 0.60

            Ipatch = I(rows,cols,:);

            counter = counter + 1;

            filename = sprintf( ...
                'ma_hehardneg_%06d.png', ...
                counter);

            imwrite( ...
                Ipatch, ...
                fullfile(imageOut,filename));

            imwrite( ...
                uint8(MAPatch), ...
                fullfile(maskOut,filename));

            createdHE = createdHE + 1;

            totalHEHard = totalHEHard + 1;
        end
    end

    %% ------------------------------------------------
    % 3. VESSEL HARD NEGATIVES
    % -------------------------------------------------

    [vr,vc] = find(vesselMask);

    createdVessel = 0;
    attemptsVessel = 0;

    while ...
            createdVessel < vesselHardNegPerImage && ...
            attemptsVessel < 1000 && ...
            ~isempty(vr)

        attemptsVessel = attemptsVessel + 1;

        k = randi(numel(vr));

        centerR = vr(k);
        centerC = vc(k);

        [rows,cols] = getPatchCoordinates( ...
            centerR, ...
            centerC, ...
            patchSize, ...
            h, ...
            w);

        MAPatch = MA(rows,cols);

        vesselPatch = vesselMask(rows,cols);

        retinaPatch = retinaMask(rows,cols);

        vesselFraction = mean(vesselPatch(:));
        retinaFraction = mean(retinaPatch(:));

        % Important:
        % Vessel patches must have a meaningful vessel amount
        % and no annotated MA.
        if nnz(MAPatch) == 0 && ...
                retinaFraction > 0.60 && ...
                vesselFraction > 0.005

            Ipatch = I(rows,cols,:);

            counter = counter + 1;

            filename = sprintf( ...
                'ma_vesselhardneg_%06d.png', ...
                counter);

            imwrite( ...
                Ipatch, ...
                fullfile(imageOut,filename));

            imwrite( ...
                uint8(MAPatch), ...
                fullfile(maskOut,filename));

            createdVessel = createdVessel + 1;

            totalVessel = totalVessel + 1;
        end
    end

    %% ------------------------------------------------
    % 4. RANDOM NORMAL RETINA NEGATIVES
    % -------------------------------------------------

    createdRandom = 0;
    attemptsRandom = 0;

    while ...
            createdRandom < randomNegPerImage && ...
            attemptsRandom < 1000

        attemptsRandom = attemptsRandom + 1;

        centerR = randi(h);
        centerC = randi(w);

        [rows,cols] = getPatchCoordinates( ...
            centerR, ...
            centerC, ...
            patchSize, ...
            h, ...
            w);

        retinaPatch = retinaMask(rows,cols);

        MAPatch = MA(rows,cols);

        retinaFraction = mean(retinaPatch(:));

        if retinaFraction > 0.70 && ...
                nnz(MAPatch) == 0

            Ipatch = I(rows,cols,:);

            counter = counter + 1;

            filename = sprintf( ...
                'ma_neg_%06d.png', ...
                counter);

            imwrite( ...
                Ipatch, ...
                fullfile(imageOut,filename));

            imwrite( ...
                uint8(MAPatch), ...
                fullfile(maskOut,filename));

            createdRandom = createdRandom + 1;

            totalRandom = totalRandom + 1;
        end
    end

    %% ------------------------------------------------
    % Progress
    % -------------------------------------------------

    fprintf( ...
        ['Processed V3 image %d / %d | ' ...
         'positive=%d | HE=%d | vessel=%d | random=%d | total=%d\n'], ...
        n, ...
        numel(trainMAIdx), ...
        createdPositive, ...
        createdHE, ...
        createdVessel, ...
        createdRandom, ...
        counter);

end

%% ---------------------------------------------------
% Final summary
% ----------------------------------------------------

fprintf('\n');
fprintf('====================================================\n');
fprintf('MICROANEURYSM V3 PATCH GENERATION COMPLETE\n');
fprintf('====================================================\n');

fprintf('\nTotal patches: %d\n',counter);

fprintf('\nPatch breakdown:\n');
fprintf('MA positive       : %d\n',totalPositive);
fprintf('HE hard negative  : %d\n',totalHEHard);
fprintf('Vessel hard       : %d\n',totalVessel);
fprintf('Random negative   : %d\n',totalRandom);

fprintf('\nOutput images:\n%s\n',imageOut);

fprintf('\nOutput masks:\n%s\n',maskOut);

%% ---------------------------------------------------
% Save split
% ----------------------------------------------------

save( ...
    'models/microaneurysmV3Split.mat', ...
    'trainMAIdx', ...
    'valMAIdx');

fprintf('\nSaved split:\n');
fprintf('models/microaneurysmV3Split.mat\n');

fprintf('====================================================\n');


%% ===================================================
% LOCAL FUNCTIONS
% ====================================================

function [rows,cols] = getPatchCoordinates( ...
    centerR, ...
    centerC, ...
    patchSize, ...
    h, ...
    w)

    r1 = round( ...
        centerR - patchSize/2);

    c1 = round( ...
        centerC - patchSize/2);

    r1 = max( ...
        1, ...
        min(r1,h-patchSize+1));

    c1 = max( ...
        1, ...
        min(c1,w-patchSize+1));

    rows = r1:r1+patchSize-1;

    cols = c1:c1+patchSize-1;

end


function probMap = predictResizedVesselMap( ...
    I, ...
    net, ...
    inputSize)

    originalH = size(I,1);
    originalW = size(I,2);

    Iresize = imresize( ...
        I, ...
        inputSize);

    scores = getNetworkScores( ...
        net, ...
        Iresize);

    probSmall = scores(:,:,2);

    probMap = imresize( ...
        probSmall, ...
        [originalH originalW], ...
        'bilinear');

end


function scores = getNetworkScores(net,I)

    if size(I,3) == 1

        I = repmat( ...
            I,[1 1 3]);

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
            'Vessel segmentation network did not return two classes.');
    end

end