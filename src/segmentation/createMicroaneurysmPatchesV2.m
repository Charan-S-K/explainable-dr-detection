%% SeeBeyond - Microaneurysm V2 Patch Generator
% Adds vessel hard-negative patches using the trained DRIVE vessel model.
%
% V1 is preserved.
%
% Patch categories:
%   1. Microaneurysm positive
%   2. Haemorrhage hard negative
%   3. Vessel hard negative
%   4. Random normal retina negative

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

rng(42);

fprintf('\n==============================================\n');
fprintf('MICROANEURYSM V2 PATCH GENERATION\n');
fprintf('==============================================\n');

%% ------------------------------------------------
% Use the SAME train/validation split as V1
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
    'datasets','IDRiD','patches','microaneurysmsV2');

imageOut = fullfile(patchRoot,'images');
maskOut  = fullfile(patchRoot,'masks');

%% ------------------------------------------------
% Clean ONLY V2 output
%% ------------------------------------------------

if exist(patchRoot,'dir')
    rmdir(patchRoot,'s');
end

mkdir(imageOut);
mkdir(maskOut);

%% ------------------------------------------------
% Load trained vessel model
%% ------------------------------------------------

fprintf('\nLoading vessel model...\n');

S = load('models/vesselNetWeighted.mat', ...
    'vesselNetWeighted');

vesselNetWeighted = S.vesselNetWeighted;

fprintf('Vessel model loaded.\n');

%% ------------------------------------------------
% Parameters
%% ------------------------------------------------

patchSize = 256;

positivePerImage  = 40;
heHardNegPerImage = 15;
vesselHardNegPerImage = 25;
randomNegPerImage = 15;

counter = 0;

%% ------------------------------------------------
% Datastore
%% ------------------------------------------------

imds = imageDatastore(imageRoot);

%% =================================================
% Process training images
%% =================================================

for n = 1:numel(trainMAIdx)

    idx = trainMAIdx(n);

    I = readimage(imds,idx);

    [~,imageID,~] = fileparts(imds.Files{idx});

    maPath = fullfile( ...
        maMaskRoot,[imageID '.png']);

    hePath = fullfile( ...
        heMaskRoot,[imageID '.png']);

    MA = imread(maPath) > 0;

    if isfile(hePath)
        HE = imread(hePath) > 0;
    else
        HE = false(size(MA));
    end

    [h,w,~] = size(I);

    %% ------------------------------------------------
    % Retina field
    %% ------------------------------------------------

    G = im2double(rgb2gray(I));

    retinaMask = G > 0.03;

    if any(retinaMask(:))
        retinaMask = bwareafilt(retinaMask,1);
        retinaMask = imfill(retinaMask,'holes');
    end

    %% =================================================
    % Generate vessel probability map
    %
    % IMPORTANT:
    % This is the same inference method used by
    % runSegmentationPipeline.m
    %% =================================================

    vesselProb = predictResizedVesselMap( ...
        I, ...
        vesselNetWeighted, ...
        [256 256]);

    vesselMask = vesselProb >= 0.50;

    vesselMask = vesselMask & retinaMask;

    vesselMask = bwareaopen(vesselMask,2);

    %% =================================================
    % 1. MICROANEURYSM POSITIVE PATCHES
    %% =================================================

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

    %% =================================================
    % 2. HAEMORRHAGE HARD NEGATIVES
    %% =================================================

    [hr,hc] = find(HE);

    createdHard = 0;
    attemptsHard = 0;

    while createdHard < heHardNegPerImage && ...
            attemptsHard < 500 && ...
            ~isempty(hr)

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

        if nnz(MAPatch) == 0 && ...
                mean(retinaPatch(:)) > 0.60

            Ipatch = I(rows,cols,:);

            counter = counter + 1;

            filename = sprintf( ...
                'ma_hehardneg_%06d.png',counter);

            imwrite(Ipatch, ...
                fullfile(imageOut,filename));

            imwrite(uint8(MAPatch), ...
                fullfile(maskOut,filename));

            createdHard = createdHard + 1;

        end

    end

    %% =================================================
    % 3. VESSEL HARD NEGATIVES
    %
    % IMPORTANT:
    % These are generated from IDRiD images using the
    % trained vessel model.
    %
    % DRIVE GT masks are NOT used here.
    %% =================================================

    [vr,vc] = find(vesselMask);

    createdVessel = 0;
    attemptsVessel = 0;

    while createdVessel < vesselHardNegPerImage && ...
            attemptsVessel < 1000 && ...
            ~isempty(vr)

        attemptsVessel = attemptsVessel + 1;

        k = randi(numel(vr));

        centerR = vr(k);
        centerC = vc(k);

        r1 = round(centerR-patchSize/2);
        c1 = round(centerC-patchSize/2);

        r1 = max(1,min(r1,h-patchSize+1));
        c1 = max(1,min(c1,w-patchSize+1));

        rows = r1:r1+patchSize-1;
        cols = c1:c1+patchSize-1;

        MAPatch = MA(rows,cols);
        vesselPatch = vesselMask(rows,cols);
        retinaPatch = retinaMask(rows,cols);

        % Require:
        %   no true MA
        %   enough retina
        %   meaningful vessel content

        vesselFraction = mean(vesselPatch(:));

        if nnz(MAPatch) == 0 && ...
                mean(retinaPatch(:)) > 0.60 && ...
                vesselFraction > 0.005

            Ipatch = I(rows,cols,:);

            counter = counter + 1;

            filename = sprintf( ...
                'ma_vesselhardneg_%06d.png',counter);

            imwrite(Ipatch, ...
                fullfile(imageOut,filename));

            imwrite(uint8(MAPatch), ...
                fullfile(maskOut,filename));

            createdVessel = createdVessel + 1;

        end

    end

    %% =================================================
    % 4. RANDOM NORMAL RETINA NEGATIVES
    %% =================================================

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

    fprintf( ...
        'Processed MA V2 image %d / %d | patches=%d\n', ...
        n,numel(trainMAIdx),counter);

end

%% =================================================
% Finish
%% =================================================

fprintf('\n==============================================\n');
fprintf('MICROANEURYSM V2 PATCH GENERATION COMPLETE\n');
fprintf('==============================================\n');

fprintf('Total patches: %d\n',counter);
fprintf('Output images: %s\n',imageOut);
fprintf('Output masks : %s\n',maskOut);

fprintf('\nPatch design per training image:\n');
fprintf('Positive MA       : %d\n',positivePerImage);
fprintf('Haemorrhage hard  : %d\n',heHardNegPerImage);
fprintf('Vessel hard       : %d\n',vesselHardNegPerImage);
fprintf('Random negatives  : %d\n',randomNegPerImage);

fprintf('==============================================\n');

save('models/microaneurysmV2Split.mat', ...
    'trainMAIdx','valMAIdx');

fprintf('\nSaved split:\n');
fprintf('models/microaneurysmV2Split.mat\n');

%% =================================================
% LOCAL FUNCTION
% Vessel probability map
%
% Same logic as runSegmentationPipeline.m
%% =================================================

function probMap = predictResizedVesselMap( ...
    I,net,inputSize)

originalH = size(I,1);
originalW = size(I,2);

Iresize = imresize(I,inputSize);

scores = getNetworkScores( ...
    net,Iresize);

probSmall = scores(:,:,2);

probMap = imresize( ...
    probSmall, ...
    [originalH originalW], ...
    'bilinear');

end

%% =================================================
% LOCAL FUNCTION
% dlnetwork prediction
%
% Same logic as runSegmentationPipeline.m
%% =================================================

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
        'Vessel segmentation network did not return two classes.');
end

end