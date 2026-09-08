function cleanMask = postprocessHardExudateMask(predMask,I)
% Clean Hard Exudate prediction.
% Removes predictions outside retina and very small false-positive regions.

predMask = logical(predMask);

% -------------------------
% Detect retinal field
% -------------------------
G = rgb2gray(I);
G = im2double(G);

retinaMask = G > 0.03;
retinaMask = bwareafilt(retinaMask,1);
retinaMask = imfill(retinaMask,'holes');

% Slightly shrink retina mask to remove bright border artifacts
retinaMask = imerode(retinaMask,strel('disk',4));

% Keep predictions only inside retina
cleanMask = predMask & retinaMask;

% -------------------------
% Remove tiny noise
% -------------------------
cleanMask = bwareaopen(cleanMask,8);

end