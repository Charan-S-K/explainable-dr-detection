function probMap = predictHardExudateProbMap(I,net)
% Produce full-resolution Hard Exudate probability map
% using overlapping 256x256 patches.

patchSize = 256;
stride = 128;

if size(I,3) == 1
    I = repmat(I,[1 1 3]);
end

[h,w,~] = size(I);

scoreMap = zeros(h,w,'single');
countMap = zeros(h,w,'single');

rowStarts = 1:stride:(h-patchSize+1);
colStarts = 1:stride:(w-patchSize+1);

% Cover bottom edge
if rowStarts(end) ~= h-patchSize+1
    rowStarts(end+1) = h-patchSize+1;
end

% Cover right edge
if colStarts(end) ~= w-patchSize+1
    colStarts(end+1) = w-patchSize+1;
end

for r = rowStarts

    rows = r:(r+patchSize-1);

    for c = colStarts

        cols = c:(c+patchSize-1);

        patch = I(rows,cols,:);

        % Explicit batch dimension
        patch4D = reshape( ...
            patch,patchSize,patchSize,3,1);

        scores = minibatchpredict( ...
            net, ...
            patch4D, ...
            MiniBatchSize=1, ...
            ExecutionEnvironment="gpu");

        if isa(scores,'dlarray')
            scores = extractdata(scores);
        end

        % Remove batch dimension
        if ndims(scores) == 4
            scores = scores(:,:,:,1);
        end

        % Channel 1 = background
        % Channel 2 = hard_exudate
        exudateScore = scores(:,:,2);

        scoreMap(rows,cols) = ...
            scoreMap(rows,cols) + single(exudateScore);

        countMap(rows,cols) = ...
            countMap(rows,cols) + 1;

    end
end

probMap = scoreMap ./ max(countMap,1);

% Retina field mask
G = im2double(rgb2gray(I));

retinaMask = G > 0.03;
retinaMask = bwareafilt(retinaMask,1);
retinaMask = imfill(retinaMask,'holes');

probMap(~retinaMask) = 0;

end