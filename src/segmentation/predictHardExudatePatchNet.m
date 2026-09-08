function predMask = predictHardExudatePatchNet(I,net)
% Sliding-window inference for Hard Exudate Patch U-Net V3

patchSize = 256;
stride = 128;

if size(I,3) == 1
    I = repmat(I,[1 1 3]);
end

[h,w,~] = size(I);

voteMap  = zeros(h,w,'single');
countMap = zeros(h,w,'single');

rowStarts = 1:stride:max(1,h-patchSize+1);
colStarts = 1:stride:max(1,w-patchSize+1);

% Make sure bottom/right edges are covered
if rowStarts(end) ~= h-patchSize+1
    rowStarts(end+1) = h-patchSize+1;
end

if colStarts(end) ~= w-patchSize+1
    colStarts(end+1) = w-patchSize+1;
end

classNames = ["background","hard_exudate"];

for r = rowStarts

    rows = r:(r+patchSize-1);

    for c = colStarts

        cols = c:(c+patchSize-1);

        patch = I(rows,cols,:);

        C = semanticseg( ...
            patch, ...
            net, ...
            Classes=classNames, ...
            ExecutionEnvironment="gpu");

        P = C == "hard_exudate";

        voteMap(rows,cols) = ...
            voteMap(rows,cols) + single(P);

        countMap(rows,cols) = ...
            countMap(rows,cols) + 1;
    end
end

probabilityMap = voteMap ./ max(countMap,1);

predMask = probabilityMap >= 0.5;

% Retina mask
G = im2double(rgb2gray(I));

retinaMask = G > 0.03;
retinaMask = bwareafilt(retinaMask,1);
retinaMask = imfill(retinaMask,'holes');

predMask = predMask & retinaMask;

% Remove isolated tiny noise
predMask = bwareaopen(predMask,4);

end