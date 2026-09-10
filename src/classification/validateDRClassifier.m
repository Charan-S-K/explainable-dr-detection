function results = validateDRClassifier()
% validateDRClassifier
% Stage 5.1 - Validate trained SeeBeyond DR classifier.

fprintf('\n========================================\n');
fprintf('SEEBEYOND STAGE 5.1 - DR CLASSIFIER VALIDATION\n');
fprintf('========================================\n');

%% Project setup

projectRoot = '/home/tharunkumar-s-v/Documents/SeeBeyond';

cd(projectRoot);
addpath(genpath(fullfile(projectRoot,'src')));

%% Load trained classifier

modelFile = fullfile( ...
    projectRoot, ...
    'models', ...
    'drClassifier.mat');

fprintf('\nLoading trained classifier...\n');

S = load(modelFile);

drClassifier = S.drClassifier;
classNames = S.classNames;
inputSize = S.inputSize;

fprintf('Model loaded successfully.\n');
fprintf('Input size: %dx%dx%d\n', ...
    inputSize(1),inputSize(2),inputSize(3));

%% Load APTOS test split

splitFile = fullfile( ...
    projectRoot, ...
    'models', ...
    'aptosStage3Split.mat');

T = load(splitFile);

testTable = T.testTable;

fprintf('\nAPTOS TEST SET\n');
fprintf('Test images: %d\n',height(testTable));

%% Create test datastore

imdsTest = imageDatastore( ...
    testTable.File, ...
    Labels=testTable.Label);

fprintf('\nTest datastore created.\n');

disp(countEachLabel(imdsTest));

%% Exact Stage 3 preprocessing

augimdsTest = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imdsTest, ...
    ColorPreprocessing="gray2rgb");

fprintf('\nPreprocessing:\n');
fprintf('  Resize: %dx%d\n', ...
    inputSize(1),inputSize(2));

fprintf('  Color: gray2rgb\n');
fprintf('  Prediction: minibatchpredict\n');

%% Prediction

fprintf('\n========================================\n');
fprintf('RUNNING TEST PREDICTION\n');
fprintf('========================================\n');

tic;

testScores = minibatchpredict( ...
    drClassifier, ...
    augimdsTest, ...
    ExecutionEnvironment="gpu");

validationTime = toc;

testScores = gather(testScores);

fprintf('\nPrediction completed.\n');
fprintf('Time: %.2f minutes\n', ...
    validationTime/60);

%% Convert scores to labels

testPred = scores2label( ...
    testScores, ...
    classNames);

testTrue = imdsTest.Labels;

%% Accuracy

testAccuracy = mean(testPred == testTrue);

fprintf('\n========================================\n');
fprintf('FINAL TEST ACCURACY\n');
fprintf('========================================\n');

fprintf('Test Accuracy: %.2f %%\n', ...
    testAccuracy*100);

%% Confusion matrix

fprintf('\n========================================\n');
fprintf('CONFUSION MATRIX\n');
fprintf('========================================\n');

confMat = confusionmat( ...
    testTrue, ...
    testPred, ...
    'Order',categorical(classNames));

disp(confMat);

%% Per-class metrics

numClasses = numel(classNames);

precision = zeros(numClasses,1);
recall = zeros(numClasses,1);
f1 = zeros(numClasses,1);

for k = 1:numClasses

    TP = confMat(k,k);

    FP = sum(confMat(:,k)) - TP;

    FN = sum(confMat(k,:)) - TP;

    precision(k) = TP / max(TP + FP,1);

    recall(k) = TP / max(TP + FN,1);

    f1(k) = ...
        2 * precision(k) * recall(k) / ...
        max(precision(k) + recall(k),eps);

end

%% Display metrics

fprintf('\n========================================\n');
fprintf('PER-CLASS METRICS\n');
fprintf('========================================\n');

for k = 1:numClasses

    fprintf('\n%s\n',classNames(k));

    fprintf('Precision : %.2f %%\n', ...
        precision(k)*100);

    fprintf('Recall    : %.2f %%\n', ...
        recall(k)*100);

    fprintf('F1 Score  : %.2f %%\n', ...
        f1(k)*100);

end

%% Macro metrics

macroPrecision = mean(precision);
macroRecall = mean(recall);
macroF1 = mean(f1);

fprintf('\n========================================\n');
fprintf('MACRO METRICS\n');
fprintf('========================================\n');

fprintf('Macro Precision : %.2f %%\n', ...
    macroPrecision*100);

fprintf('Macro Recall    : %.2f %%\n', ...
    macroRecall*100);

fprintf('Macro F1        : %.2f %%\n', ...
    macroF1*100);

%% Results structure

results = struct();

results.testAccuracy = testAccuracy;
results.precision = precision;
results.recall = recall;
results.f1 = f1;

results.macroPrecision = macroPrecision;
results.macroRecall = macroRecall;
results.macroF1 = macroF1;

results.confMat = confMat;
results.classNames = classNames;

results.numTestImages = height(testTable);
results.validationTime = validationTime;

results.preprocessing = ...
    '224x224 augmentedImageDatastore with gray2rgb';

%% Save result

resultFile = fullfile( ...
    projectRoot, ...
    'models', ...
    'drClassifierValidation.mat');

save(resultFile, ...
    'results', ...
    '-v7.3');

fprintf('\n========================================\n');
fprintf('VALIDATION RESULT SAVED\n');
fprintf('========================================\n');

fprintf('%s\n',resultFile);

fprintf('\n========================================\n');
fprintf('STAGE 5.1 COMPLETE\n');
fprintf('========================================\n');

end