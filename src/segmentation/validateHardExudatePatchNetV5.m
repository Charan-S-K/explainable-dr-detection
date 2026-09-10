function validateHardExudatePatchNetV5()
% validateHardExudatePatchNetV5
% Validate Hard Exudate V5 patch model on the held-out IDRiD images.

clc;

projectRoot = '/home/tharunkumar-s-v/Documents/SeeBeyond';

imageRoot = fullfile(projectRoot, ...
    'datasets','IDRiD','Segmentation','A. Segmentation', ...
    '1. Original Images','a. Training Set');

maskRoot = fullfile(projectRoot, ...
    'datasets','IDRiD','processed_masks','hard_exudates');

modelPath = fullfile(projectRoot, ...
    'models','hardExudatePatchNetV5.mat');

S = load(modelPath,'hardExudatePatchNetV5');
net = S.hardExudatePatchNetV5;

% Same validation split used by Hard Exudate V2
Ssplit = load(fullfile(projectRoot, ...
    'models','hardExudateNetV2.mat'), ...
    'valEXIdx');

valIDs = Ssplit.valEXIdx;

patchSize = 256;
threshold = 0.50;

diceValues = [];
iouValues = [];
sensitivityValues = [];
specificityValues = [];
precisionValues = [];

fprintf('\n=============================================\n');
fprintf(' HARD EXUDATE V5 VALIDATION\n');
fprintf('=============================================\n');

fprintf('Validation images: %d\n',numel(valIDs));
fprintf('Threshold: %.2f\n',threshold);

for k = 1:numel(valIDs)

    id = valIDs(k);

    imageName = sprintf('IDRiD_%02d.jpg',id);
    maskName  = sprintf('IDRiD_%02d.png',id);

    imagePath = fullfile(imageRoot,imageName);
    maskPath  = fullfile(maskRoot,maskName);

    if ~isfile(imagePath)
        warning('Missing image: %s',imagePath);
        continue;
    end

    if ~isfile(maskPath)
        warning('Missing mask: %s',maskPath);
        continue;
    end

    I = imread(imagePath);

    if size(I,3) == 1
        I = repmat(I,[1 1 3]);
    end

    GT = imread(maskPath) > 0;

    % Generate validation patches using deterministic sampling.
    rng(42 + id);

    [H,W,~] = size(I);

    retina = rgb2gray(I);
    retina = retina > 8;
    retina = bwareafilt(retina,1);
    retina = imfill(retina,'holes');

    lesionPixels = find(GT & retina);

    if isempty(lesionPixels)
        fprintf('IDRiD_%02d: no lesion pixels\n',id);
        continue;
    end

    [ly,lx] = ind2sub([H W],lesionPixels);

    numPositive = min(20,numel(lesionPixels));

    positiveIdx = randperm(numel(lesionPixels),numPositive);

    TP = 0;
    TN = 0;
    FP = 0;
    FN = 0;

    for p = 1:numPositive

        cx = lx(positiveIdx(p));
        cy = ly(positiveIdx(p));

        x1 = cx - floor(patchSize/2);
        y1 = cy - floor(patchSize/2);

        x1 = max(1,min(x1,W-patchSize+1));
        y1 = max(1,min(y1,H-patchSize+1));

        x2 = x1 + patchSize - 1;
        y2 = y1 + patchSize - 1;

        patch = I(y1:y2,x1:x2,:);

        gtPatch = GT(y1:y2,x1:x2);

        predPatch = predictHardExudatePatchV5(net,patch);

        predMask = predPatch >= threshold;

        TP = TP + sum(predMask(:) & gtPatch(:));
        TN = TN + sum(~predMask(:) & ~gtPatch(:));
        FP = FP + sum(predMask(:) & ~gtPatch(:));
        FN = FN + sum(~predMask(:) & gtPatch(:));

    end

    % Negative patches
    negativeCount = 20;

    for p = 1:negativeCount

        for attempt = 1:100

            cx = randi([floor(patchSize/2),W-floor(patchSize/2)]);
            cy = randi([floor(patchSize/2),H-floor(patchSize/2)]);

            x1 = cx - floor(patchSize/2);
            y1 = cy - floor(patchSize/2);

            x1 = max(1,min(x1,W-patchSize+1));
            y1 = max(1,min(y1,H-patchSize+1));

            x2 = x1 + patchSize - 1;
            y2 = y1 + patchSize - 1;

            gtPatch = GT(y1:y2,x1:x2);

            if nnz(gtPatch) == 0
                break;
            end

        end

        patch = I(y1:y2,x1:x2,:);

        predPatch = predictHardExudatePatchV5(net,patch);

        predMask = predPatch >= threshold;

        TP = TP + sum(predMask(:) & gtPatch(:));
        TN = TN + sum(~predMask(:) & ~gtPatch(:));
        FP = FP + sum(predMask(:) & ~gtPatch(:));
        FN = FN + sum(~predMask(:) & gtPatch(:));

    end

    dice = (2*TP) / max(2*TP + FP + FN,1);

    iou = TP / max(TP + FP + FN,1);

    sensitivity = TP / max(TP + FN,1);

    specificity = TN / max(TN + FP,1);

    precision = TP / max(TP + FP,1);

    diceValues(end+1) = dice;
    iouValues(end+1) = iou;
    sensitivityValues(end+1) = sensitivity;
    specificityValues(end+1) = specificity;
    precisionValues(end+1) = precision;

    fprintf(['IDRiD_%02d | Dice %.4f | IoU %.4f | ' ...
        'Sens %.4f | Spec %.4f | Prec %.4f\n'], ...
        id,dice,iou,sensitivity,specificity,precision);

end

fprintf('\n=============================================\n');
fprintf(' HARD EXUDATE V5 RESULTS\n');
fprintf('=============================================\n');

meanDice = mean(diceValues);
meanIoU = mean(iouValues);
meanSensitivity = mean(sensitivityValues);
meanSpecificity = mean(specificityValues);
meanPrecision = mean(precisionValues);

fprintf('Mean Dice        : %.4f (%.2f%%)\n', ...
    meanDice,100*meanDice);

fprintf('Mean IoU         : %.4f (%.2f%%)\n', ...
    meanIoU,100*meanIoU);

fprintf('Mean Sensitivity : %.4f (%.2f%%)\n', ...
    meanSensitivity,100*meanSensitivity);

fprintf('Mean Specificity : %.4f (%.2f%%)\n', ...
    meanSpecificity,100*meanSpecificity);

fprintf('Mean Precision   : %.4f (%.2f%%)\n', ...
    meanPrecision,100*meanPrecision);

fprintf('\nBaseline Hard Exudate V2 + post-processing:\n');
fprintf('Dice = 0.1892 (18.92%%)\n');

if meanDice > 0.1892
    fprintf('\nRESULT: V5 IMPROVED over the previous baseline.\n');
else
    fprintf('\nRESULT: V5 did NOT improve over the previous baseline.\n');
end

fprintf('=============================================\n');

save(fullfile(projectRoot,'results', ...
    'hardExudateV5Validation.mat'), ...
    'meanDice','meanIoU','meanSensitivity', ...
    'meanSpecificity','meanPrecision', ...
    'diceValues','iouValues', ...
    'sensitivityValues','specificityValues', ...
    'precisionValues','valIDs','threshold');

end


function prob = predictHardExudatePatchV5(net,I)

I = im2single(I);

if size(I,3) == 1
    I = repmat(I,[1 1 3]);
end

% Network expects 256x256x3
I = imresize(I,[256 256]);

X = dlarray(I,'SSCB');

if canUseGPU
    X = gpuArray(X);
end

Y = predict(net,X);

Y = gather(extractdata(Y));

if ndims(Y) == 4
    Y = Y(:,:,2,1);
else
    Y = Y(:,:,2);
end

prob = Y;

end