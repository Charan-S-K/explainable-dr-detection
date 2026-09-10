function result = predictDRSeverity(I, drClassifier)
% predictDRSeverity
% Predict diabetic retinopathy severity from one fundus image.
%
% Uses the SAME preprocessing approach as the Stage 3
% validation pipeline.

%% Configuration

inputSize = [224 224 3];

classNames = [
    "No_DR"
    "Mild"
    "Moderate"
    "Severe"
    "Proliferative_DR"
];

%% Validate image

if isempty(I)
    error('Input image is empty.');
end

if size(I,3) == 1

    I = repmat(I,1,1,3);

elseif size(I,3) ~= 3

    error('Input image must be grayscale or RGB.');

end

%% Convert to uint8

if ~isa(I,'uint8')
    I = im2uint8(I);
end

%% Create temporary image

tempDir = tempname;
mkdir(tempDir);

tempFile = fullfile(tempDir,'input.png');

imwrite(I,tempFile);

%% Create datastore

imds = imageDatastore(tempFile);

%% IMPORTANT:
% Match the Stage 3 validation pipeline exactly.
%
% Do NOT resize the image manually here.

augimds = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imds, ...
    ColorPreprocessing="gray2rgb");

%% Predict

scores = minibatchpredict( ...
    drClassifier, ...
    augimds, ...
    ExecutionEnvironment="gpu");

scores = gather(scores);

scores = squeeze(scores);

%% Convert network output to probabilities

% If outputs are already probabilities, use them directly.
% Otherwise apply softmax.

if all(scores >= 0) && abs(sum(scores) - 1) < 1e-4

    probabilities = scores;

else

    scores = scores - max(scores);

    probabilities = exp(scores);

    probabilities = probabilities ./ sum(probabilities);

end

%% Find prediction

[confidence,index] = max(probabilities);

predictedClass = classNames(index);

%% Store result

result = struct();

result.class = predictedClass;

result.classIndex = index - 1;

result.confidence = confidence;

result.probabilities = probabilities;

result.classNames = classNames;

%% Display result

fprintf('\n');
fprintf('========================================\n');
fprintf('SEE BEYOND DR CLASSIFICATION\n');
fprintf('========================================\n');

fprintf('Predicted DR : %s\n',predictedClass);

fprintf('Confidence   : %.2f %%\n', ...
    confidence * 100);

fprintf('\nClass probabilities:\n');

for k = 1:numel(classNames)

    fprintf('  %-20s : %.2f %%\n', ...
        classNames(k), ...
        probabilities(k) * 100);

end

fprintf('========================================\n');

%% Cleanup

if isfile(tempFile)
    delete(tempFile);
end

if isfolder(tempDir)
    rmdir(tempDir);
end

end