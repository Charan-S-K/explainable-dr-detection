function prepareIDRiDLesionMasks(imageDir, sourceMaskDir, outputDir, suffix)
% prepareIDRiDLesionMasks
% Creates one binary lesion mask for every IDRiD training image.
% If a lesion annotation does not exist, creates an empty mask.

if ~exist(outputDir,'dir')
    mkdir(outputDir);
end

imds = imageDatastore(imageDir);

for i = 1:numel(imds.Files)

    imagePath = imds.Files{i};

    [~,imageID,~] = fileparts(imagePath);

    I = imread(imagePath);
    [h,w,~] = size(I);

    sourcePath = fullfile( ...
        sourceMaskDir, ...
        imageID + "_" + suffix + ".tif");

    if isfile(sourcePath)
        M = imread(sourcePath);
        M = M > 0;
    else
        % No lesion annotation = lesion absent
        M = false(h,w);
    end

    outputPath = fullfile( ...
        outputDir, ...
        imageID + ".png");

    imwrite(M,outputPath);

end

fprintf('Prepared %d masks in:\n%s\n', ...
    numel(imds.Files),outputDir);
end