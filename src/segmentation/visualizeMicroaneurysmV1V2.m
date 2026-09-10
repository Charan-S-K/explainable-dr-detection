%% SeeBeyond - Microaneurysm V1 vs V2 Visualization

clear;
clc;
close all;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

%% Load models
S1 = load('models/microaneurysmPatchNet.mat', ...
    'microaneurysmPatchNet');

S2 = load('models/microaneurysmPatchNetV2.mat', ...
    'microaneurysmPatchNetV2');

netV1 = S1.microaneurysmPatchNet;
netV2 = S2.microaneurysmPatchNetV2;

%% Select validation image
imageID = 'IDRiD_36';

idridRoot = fullfile(pwd, ...
    'datasets','IDRiD','Segmentation','A. Segmentation');

imageRoot = fullfile(idridRoot, ...
    '1. Original Images','a. Training Set');

maskRoot = fullfile(pwd, ...
    'datasets','IDRiD','processed_masks','microaneurysms');

imagePath = fullfile(imageRoot,[imageID '.jpg']);
maskPath  = fullfile(maskRoot,[imageID '.png']);

I = imread(imagePath);
GT = imread(maskPath) > 0;

fprintf('\n=============================================\n');
fprintf('MICROANEURYSM V1 vs V2 VISUALIZATION\n');
fprintf('=============================================\n');
fprintf('Image: %s\n',imageID);

%% Find a positive region
[rr,cc] = find(GT);

if isempty(rr)
    error('No microaneurysm pixels found in %s.',imageID);
end

rng(42);
k = randi(numel(rr));

centerR = rr(k);
centerC = cc(k);

patchSize = 256;

r1 = round(centerR - patchSize/2);
c1 = round(centerC - patchSize/2);

r1 = max(1,min(r1,size(I,1)-patchSize+1));
c1 = max(1,min(c1,size(I,2)-patchSize+1));

rows = r1:r1+patchSize-1;
cols = c1:c1+patchSize-1;

Ipatch = I(rows,cols,:);
GTpatch = GT(rows,cols);

%% Predict V1
[predV1,scoreV1] = predictMAPatch(netV1,Ipatch);

%% Predict V2
[predV2,scoreV2] = predictMAPatch(netV2,Ipatch);

%% Calculate patch metrics
diceV1 = 2*nnz(predV1 & GTpatch) / ...
    max(2*nnz(predV1 & GTpatch) + ...
    nnz(predV1 & ~GTpatch) + ...
    nnz(~predV1 & GTpatch),eps);

diceV2 = 2*nnz(predV2 & GTpatch) / ...
    max(2*nnz(predV2 & GTpatch) + ...
    nnz(predV2 & ~GTpatch) + ...
    nnz(~predV2 & GTpatch),eps);

fprintf('\nPatch Dice:\n');
fprintf('V1 = %.4f\n',diceV1);
fprintf('V2 = %.4f\n',diceV2);

fprintf('\nPredicted pixels:\n');
fprintf('V1 = %d\n',nnz(predV1));
fprintf('V2 = %d\n',nnz(predV2));
fprintf('GT = %d\n',nnz(GTpatch));

%% Display comparison
figure('Name','Microaneurysm V1 vs V2', ...
    'Color','w', ...
    'Position',[100 100 1400 800]);

subplot(2,3,1);
imshow(Ipatch);
title('Original patch');

subplot(2,3,2);
imshow(GTpatch);
title('Ground Truth');

subplot(2,3,3);
imshow(predV1);
title(sprintf('V1 prediction | Dice %.3f',diceV1));

subplot(2,3,4);
imshow(predV2);
title(sprintf('V2 prediction | Dice %.3f',diceV2));

subplot(2,3,5);
imshow(scoreV1,[]);
title('V1 probability');

subplot(2,3,6);
imshow(scoreV2,[]);
title('V2 probability');

sgtitle(sprintf('Microaneurysm: %s',imageID));

%% Save screenshot automatically
if ~exist('results','dir')
    mkdir('results');
end

saveas(gcf, ...
    'results/microaneurysmV1V2Comparison.png');

fprintf('\nSaved:\n');
fprintf('results/microaneurysmV1V2Comparison.png\n');
fprintf('=============================================\n');


%% Local prediction function
function [mask,scoreMap] = predictMAPatch(net,Ipatch)

    X = single(Ipatch);

    Y = predict(net,X);

    Y = extractdata(Y);
    Y = squeeze(Y);

    if ndims(Y) ~= 3 || size(Y,3) ~= 2
        error('Unexpected Microaneurysm network output size.');
    end

    scoreMap = Y(:,:,2);

    mask = scoreMap >= 0.50;
end