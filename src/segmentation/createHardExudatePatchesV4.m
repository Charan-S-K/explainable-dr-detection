%% SeeBeyond - Hard Exudate V4 Hard-Negative Patch Mining

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

rng(42);

%% Load split + V3 model
load('models/hardExudateNetV2.mat','trainEXIdx');
load('models/hardExudatePatchNetV3.mat','hardExudatePatchNetV3');

%% Paths
idridRoot = fullfile(pwd, ...
    'datasets','IDRiD','Segmentation','A. Segmentation');

imageRoot = fullfile(idridRoot, ...
    '1. Original Images','a. Training Set');

maskRoot = fullfile(pwd, ...
    'datasets','IDRiD','processed_masks','hard_exudates');

v4Root = fullfile(pwd, ...
    'datasets','IDRiD','patches','hard_exudates_v4');

imageOut = fullfile(v4Root,'images');
maskOut  = fullfile(v4Root,'masks');

if exist(v4Root,'dir')
    rmdir(v4Root,'s');
end

mkdir(imageOut);
mkdir(maskOut);

imds = imageDatastore(imageRoot);

patchSize = 256;

positivePerImage = 25;
hardNegativePerImage = 20;
randomNegativePerImage = 10;

counter = 0;

fprintf('\n===== HARD EXUDATE V4 PATCH MINING =====\n');

for n = 1:numel(trainEXIdx)

    idxImage = trainEXIdx(n);

    I = readimage(imds,idxImage);

    [~,imageID,~] = fileparts(imds.Files{idxImage});

    M = imread(fullfile(maskRoot,imageID + ".png")) > 0;

    [h,w,~] = size(I);

    %% Retina mask
    G = im2double(rgb2gray(I));

    retinaMask = G > 0.03;
    retinaMask = bwareafilt(retinaMask,1);
    retinaMask = imfill(retinaMask,'holes');

    %% -----------------------------------------
    % 1. POSITIVE LESION PATCHES
    %% -----------------------------------------

    [rr,cc] = find(M);

    if ~isempty(rr)

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

            counter = counter + 1;

            name = sprintf('pos_%06d.png',counter);

            imwrite(I(rows,cols,:), ...
                fullfile(imageOut,name));

            imwrite(uint8(M(rows,cols)), ...
                fullfile(maskOut,name));
        end
    end

    %% -----------------------------------------
    % 2. V3 HARD FALSE-POSITIVE PATCHES
    %% -----------------------------------------

    fprintf('Image %d/%d - mining V3 hard negatives...\n', ...
        n,numel(trainEXIdx));

    prob = predictHardExudateProbMap( ...
        I,hardExudatePatchNetV3);

    falsePositiveRegion = ...
        prob >= 0.80 & ...
        ~M & ...
        retinaMask;

    [fr,fc] = find(falsePositiveRegion);

    if ~isempty(fr)

        for p = 1:hardNegativePerImage

            k = randi(numel(fr));

            centerR = fr(k);
            centerC = fc(k);

            r1 = round(centerR-patchSize/2);
            c1 = round(centerC-patchSize/2);

            r1 = max(1,min(r1,h-patchSize+1));
            c1 = max(1,min(c1,w-patchSize+1));

            rows = r1:r1+patchSize-1;
            cols = c1:c1+patchSize-1;

            % Skip if accidentally contains substantial real lesion
            MPatch = M(rows,cols);

            if nnz(MPatch) > 20
                continue
            end

            counter = counter + 1;

            name = sprintf('hardneg_%06d.png',counter);

            imwrite(I(rows,cols,:), ...
                fullfile(imageOut,name));

            imwrite(uint8(MPatch), ...
                fullfile(maskOut,name));
        end
    end

    %% -----------------------------------------
    % 3. RANDOM NORMAL RETINA PATCHES
    %% -----------------------------------------

    created = 0;
    attempts = 0;

    while created < randomNegativePerImage && attempts < 500

        attempts = attempts + 1;

        centerR = randi(h);
        centerC = randi(w);

        r1 = round(centerR-patchSize/2);
        c1 = round(centerC-patchSize/2);

        r1 = max(1,min(r1,h-patchSize+1));
        c1 = max(1,min(c1,w-patchSize+1));

        rows = r1:r1+patchSize-1;
        cols = c1:c1+patchSize-1;

        RP = retinaMask(rows,cols);
        MP = M(rows,cols);

        if mean(RP(:)) > 0.70 && nnz(MP) == 0

            counter = counter + 1;

            name = sprintf('neg_%06d.png',counter);

            imwrite(I(rows,cols,:), ...
                fullfile(imageOut,name));

            imwrite(uint8(MP), ...
                fullfile(maskOut,name));

            created = created + 1;
        end
    end

end

fprintf('\n===============================\n');
fprintf('V4 patches created: %d\n',counter);
fprintf('===============================\n');