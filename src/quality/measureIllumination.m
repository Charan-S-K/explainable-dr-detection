function [brightnessScore, illuminationStatus] = measureIllumination(I)
% measureIllumination - Evaluate illumination of a fundus image
%
% Output:
%   brightnessScore     - Mean retinal brightness from 0 to 1
%   illuminationStatus  - "Too Dark", "Acceptable", or "Too Bright"

% Convert image to grayscale
if size(I,3) == 3
    G = rgb2gray(I);
else
    G = I;
end

G = im2double(G);

% Ignore black background around the retina
retinaMask = G > 0.05;

% Keep largest connected retinal region
retinaMask = bwareafilt(retinaMask,1);

retinalPixels = G(retinaMask);

if isempty(retinalPixels)
    brightnessScore = 0;
    illuminationStatus = "Invalid Image";
    return;
end

% Average brightness only inside retinal region
brightnessScore = mean(retinalPixels);

% Initial thresholds - later calibrated using datasets
if brightnessScore < 0.20
    illuminationStatus = "Too Dark";

elseif brightnessScore > 0.75
    illuminationStatus = "Too Bright";

else
    illuminationStatus = "Acceptable";
end
end