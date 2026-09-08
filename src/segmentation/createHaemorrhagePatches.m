%% SeeBeyond - Haemorrhage Patch Generator

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

rng(42);

load('models/haemorrhageNet.mat','trainHEIdx');

idridRoot = fullfile(pwd, ...
    'datasets','IDRiD','Segmentation','A. Segmentation');

imageRoot = fullfile(idridRoot, ...
    '1. Original Images','a. Training Set');

maskRoot = fullfile(pwd, ...
    'datasets','IDRiD','processed_masks','haemorrhages');

patchRoot = fullfile(pwd, ...
    'datasets','IDRiD','patches','haemorrhages');

imageOut = fullfile(patchRoot,'images');
maskOut  = fullfile(patchRoot,'masks');

if exist(patchRoot,'dir')
    rmdir(patchRoot,'s');
end

mkdir(imageOut);
mkdir(maskOut);

imds = imageDatastore(imageRoot);

patchSize = 256;

positivePerImage = 30;
negativePerImage = 20;

counter = 0;

fprintf('\n===== CREATING HAEMORRHAGE PATCHES =====\n');

for n = 1:numel(trainHEIdx)

    idx = trainHEIdx(n);

    I = readimage(imds,idx);

    [~,imageID,~] = fileparts(imds.Files{idx});

    M = imread(fullfile(maskRoot,imageID + ".png")) > 0;

    [h,w,~] = size(I);

    G = im2double(rgb2gray(I));

    retinaMask = G > 0.03;
    retinaMask = bwareafilt(retinaMask,1);
    retinaMask = imfill(retinaMask,'holes');

    %% Positive patches

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

            name = sprintf('he_pos_%06d.png',counter);

            imwrite(I(rows,cols,:), ...
                fullfile(imageOut,name));

            imwrite(uint8(M(rows,cols)), ...
                fullfile(maskOut,name));

        end
    end

    %% Negative retinal patches

    created = 0;
    attempts = 0;

    while created < negativePerImage && attempts < 500

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

            name = sprintf('he_neg_%06d.png',counter);

            imwrite(I(rows,cols,:), ...
                fullfile(imageOut,name));

            imwrite(uint8(MP), ...
                fullfile(maskOut,name));

            created = created + 1;
        end
    end

    fprintf('Processed %d / %d\n',n,numel(trainHEIdx));

end

fprintf('\nHaemorrhage patches created: %d\n',counter);