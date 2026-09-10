function probMap = predictHaemorrhageProbMapV2(I,net)
% SeeBeyond - Haemorrhage V2 full-resolution probability map
%
% Sliding-window inference:
% Patch size = 256
% Stride     = 128

patchSize = 256;
stride = 128;

if size(I,3) == 1
    I = repmat(I,[1 1 3]);
end

[h,w,~] = size(I);

scoreMap = zeros(h,w,'single');
countMap = zeros(h,w,'single');

%% Patch starting positions

rowStarts = 1:stride:(h-patchSize+1);
colStarts = 1:stride:(w-patchSize+1);

% Ensure bottom edge is covered
if rowStarts(end) ~= h-patchSize+1
    rowStarts(end+1) = h-patchSize+1;
end

% Ensure right edge is covered
if colStarts(end) ~= w-patchSize+1
    colStarts(end+1) = w-patchSize+1;
end

%% Sliding-window prediction

for r = rowStarts

    rows = r:r+patchSize-1;

    for c = colStarts

        cols = c:c+patchSize-1;

        patch = I(rows,cols,:);

        X = reshape( ...
            patch, ...
            patchSize,patchSize,3,1);

        scores = minibatchpredict( ...
            net, ...
            X, ...
            MiniBatchSize=1, ...
            ExecutionEnvironment="gpu");

        if isa(scores,'dlarray')
            scores = extractdata(scores);
        end

        if isa(scores,'gpuArray')
            scores = gather(scores);
        end

        if ndims(scores) == 4
            scores = scores(:,:,:,1);
        end

        % Channel 1 = background
        % Channel 2 = haemorrhage
        lesionScore = scores(:,:,2);

        scoreMap(rows,cols) = ...
            scoreMap(rows,cols) + single(lesionScore);

        countMap(rows,cols) = ...
            countMap(rows,cols) + 1;

    end
end

%% Average overlapping predictions

probMap = ...
    scoreMap ./ max(countMap,1);

%% Retina masking

G = im2double(rgb2gray(I));

retinaMask = G > 0.03;

if any(retinaMask(:))

    retinaMask = bwareafilt(retinaMask,1);
    retinaMask = imfill(retinaMask,'holes');

end

probMap(~retinaMask) = 0;

end