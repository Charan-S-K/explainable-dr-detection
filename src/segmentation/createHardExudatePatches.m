%% SeeBeyond - Create Hard Exudate Training Patches

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

rng(42);

%% Load the SAME train/validation split used for V2

load('models/hardExudateNetV2.mat','trainEXIdx','valEXIdx');

%% Paths

idridRoot = fullfile(pwd, ...
    'datasets','IDRiD','Segmentation','A. Segmentation');

imageRoot = fullfile(idridRoot, ...
    '1. Original Images','a. Training Set');

maskRoot = fullfile(pwd, ...
    'datasets','IDRiD','processed_masks','hard_exudates');

patchRoot = fullfile(pwd, ...
    'datasets','IDRiD','patches','hard_exudates');

patchImageDir = fullfile(patchRoot,'images');
patchMaskDir  = fullfile(patchRoot,'masks');

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

% Generate more positive than negative patches
positivePatchesPerImage = 25;
negativePatchesPerImage = 10;

patchCounter = 0;

fprintf('\n===== Creating Hard Exudate Patches =====\n');

%% Only use the 43 TRAINING images

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

    %% Retina mask for selecting useful background patches

    G = im2double(rgb2gray(I));

    retinaMask = G > 0.03;
    retinaMask = bwareafilt(retinaMask,1);
    retinaMask = imfill(retinaMask,'holes');

    %% ----------------------------------
    % POSITIVE PATCHES
    %% ----------------------------------

    [lesionRows,lesionCols] = find(M);

    if ~isempty(lesionRows)

        for p = 1:positivePatchesPerImage

            pick = randi(numel(lesionRows));

            centerR = lesionRows(pick);
            centerC = lesionCols(pick);

            % Small random shift so patches are not identical
            centerR = centerR + randi([-64 64]);
            centerC = centerC + randi([-64 64]);

            r1 = round(centerR - patchSize/2);
            c1 = round(centerC - patchSize/2);

            r1 = max(1,min(r1,h-patchSize+1));
            c1 = max(1,min(c1,w-patchSize+1));

            rows = r1:(r1+patchSize-1);
            cols = c1:(c1+patchSize-1);

            Ipatch = I(rows,cols,:);
            Mpatch = M(rows,cols);

            patchCounter = patchCounter + 1;

            filename = sprintf('patch_%05d.png',patchCounter);

            imwrite(Ipatch, ...
                fullfile(patchImageDir,filename));

            imwrite(uint8(Mpatch), ...
                fullfile(patchMaskDir,filename));

        end
    end

    %% ----------------------------------
    % NEGATIVE PATCHES
    %% ----------------------------------

    negativeCreated = 0;
    attempts = 0;

    while negativeCreated < negativePatchesPerImage && attempts < 500

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

        % Background patch must be mostly inside retina
        % and contain no exudates
        if mean(retinaPatch(:)) > 0.60 && nnz(Mpatch) == 0

            Ipatch = I(rows,cols,:);

            patchCounter = patchCounter + 1;

            filename = sprintf('patch_%05d.png',patchCounter);

            imwrite(Ipatch, ...
                fullfile(patchImageDir,filename));

            imwrite(uint8(Mpatch), ...
                fullfile(patchMaskDir,filename));

            negativeCreated = negativeCreated + 1;
        end
    end

    fprintf('Processed training image %d / %d\n', ...
        n,numel(trainEXIdx));

end

fprintf('\n======================================\n');
fprintf('Total patches created: %d\n',patchCounter);
fprintf('Images: %s\n',patchImageDir);
fprintf('Masks : %s\n',patchMaskDir);
fprintf('======================================\n');