%% SeeBeyond - Haemorrhage U-Net V4
% Lesion-only Dice + Weighted CE

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

rng(42);

fprintf('\n========================================\n');
fprintf('HAEMORRHAGE PATCH U-NET V4\n');
fprintf('========================================\n');

%% Split

load('models/haemorrhageV2Split.mat', ...
    'trainHEIdx','valHEIdx');

%% Existing V2 patches

patchRoot = fullfile(pwd, ...
    'datasets','IDRiD','patches','haemorrhages_v2');

imageDir = fullfile(patchRoot,'images');
maskDir  = fullfile(patchRoot,'masks');

%% Datastores

imds = imageDatastore(imageDir);

classNames = [
    "background"
    "haemorrhage"
    ];

pxds = pixelLabelDatastore( ...
    maskDir, ...
    classNames, ...
    [0 1], ...
    FileExtensions=".png");

fprintf('Images : %d\n',numel(imds.Files));
fprintf('Masks  : %d\n',numel(pxds.Files));

%% Data

ds = combine(imds,pxds);
ds = transform(ds,@augmentLesionData);

sample = preview(ds);

fprintf('Image size:\n');
disp(size(sample{1}));

fprintf('Undefined pixels : %d\n', ...
    sum(isundefined(sample{2}(:))));

%% Network

net = unet( ...
    [256 256 3], ...
    2, ...
    EncoderDepth=3, ...
    NumFirstEncoderFilters=16);

%% Training

options = trainingOptions("adam", ...
    InitialLearnRate=5e-5, ...
    MaxEpochs=30, ...
    MiniBatchSize=4, ...
    Shuffle="every-epoch", ...
    ExecutionEnvironment="gpu", ...
    Plots="training-progress", ...
    Verbose=true);

fprintf('\nStarting V4 training...\n');

haemorrhagePatchNetV4 = trainnet( ...
    ds, ...
    net, ...
    @haemorrhageV4Loss, ...
    options);

%% Save separately

save('models/haemorrhagePatchNetV4.mat', ...
    'haemorrhagePatchNetV4', ...
    'trainHEIdx', ...
    'valHEIdx');

fprintf('\n========================================\n');
fprintf('HAEMORRHAGE V4 TRAINING COMPLETE\n');
fprintf('========================================\n');

fprintf('Saved: models/haemorrhagePatchNetV4.mat\n');