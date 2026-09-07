function [fovScore, fovStatus, details] = evaluateFOV(I)
% evaluateFOV - Evaluate retinal field-of-view quality
%
% Outputs:
%   fovScore  - overall score from 0 to 1
%   fovStatus - "Acceptable", "Borderline", or "Poor"
%   details   - structure with measurements

if size(I,3) == 3
    G = rgb2gray(I);
else
    G = I;
end

G = im2double(G);

[h,w] = size(G);

% Detect visible retinal region
retinaMask = G > 0.05;

% Remove small isolated regions
retinaMask = bwareaopen(retinaMask, round(0.001*h*w));

% Keep largest connected component
retinaMask = bwareafilt(retinaMask,1);

% Fill holes
retinaMask = imfill(retinaMask,'holes');

stats = regionprops(retinaMask, ...
    'Area','Centroid','BoundingBox');

if isempty(stats)
    fovScore = 0;
    fovStatus = "Poor";

    details.coverage = 0;
    details.centerOffset = 1;
    details.centroid = [NaN NaN];

    return;
end

% -------------------------------
% 1. Retinal area coverage
% -------------------------------

coverage = stats.Area / (h*w);

% Score becomes high when substantial retina is visible
coverageScore = min(coverage / 0.45, 1);

% -------------------------------
% 2. Centering
% -------------------------------

imageCenter = [w/2 h/2];
retinaCenter = stats.Centroid;

dx = (retinaCenter(1)-imageCenter(1)) / (w/2);
dy = (retinaCenter(2)-imageCenter(2)) / (h/2);

centerOffset = sqrt(dx^2 + dy^2);

centerScore = max(0, 1-centerOffset);

% -------------------------------
% Overall FOV score
% -------------------------------

fovScore = ...
    0.65 * coverageScore + ...
    0.35 * centerScore;

% Provisional thresholds
if fovScore >= 0.80
    fovStatus = "Acceptable";

elseif fovScore >= 0.60
    fovStatus = "Borderline";

else
    fovStatus = "Poor";
end

% Return measurements
details.coverage = coverage;
details.centerOffset = centerOffset;
details.centroid = retinaCenter;
details.mask = retinaMask;
end
