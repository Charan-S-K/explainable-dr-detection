function blurScore = measureBlur(I)
% measureBlur - Estimate sharpness of a retinal fundus image.
%
% Higher blurScore = sharper image
% Lower blurScore  = blurrier image
%
% Input:
%   I - RGB or grayscale retinal image
%
% Output:
%   blurScore - variance of Laplacian inside retinal region

% Convert to grayscale
if size(I,3) == 3
    G = rgb2gray(I);
else
    G = I;
end

G = im2double(G);

% Create approximate retinal region mask
% Ignore black background surrounding fundus image
retinaMask = G > 0.05;

% Keep largest connected region
retinaMask = bwareafilt(retinaMask,1);

% Remove strong circular border from calculation
retinaMask = imerode(retinaMask,ones(15));

% Laplacian filter
H = [0 1 0;
    1 -4 1;
    0 1 0];

L = imfilter(G,H,'replicate');

% Use only retinal pixels
values = L(retinaMask);

if isempty(values)
    blurScore = 0;
else
    blurScore = var(values);
end
end