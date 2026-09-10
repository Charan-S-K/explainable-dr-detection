function Iout = prepareFundusForVesselNet(I)

if size(I,3) ~= 3
    I = repmat(I,[1 1 3]);
end

% Detect retinal region
G = rgb2gray(I);
G = im2double(G);

mask = G > 0.04;
mask = bwareafilt(mask,1);
mask = imfill(mask,'holes');

stats = regionprops(mask,'BoundingBox');

if ~isempty(stats)
    bbox = stats(1).BoundingBox;
    I = imcrop(I,bbox);
end

% Pad to square without stretching
[h,w,~] = size(I);
side = max(h,w);

padTop    = floor((side-h)/2);
padBottom = ceil((side-h)/2);
padLeft   = floor((side-w)/2);
padRight  = ceil((side-w)/2);

I = padarray(I,[padTop padLeft],0,'pre');
I = padarray(I,[padBottom padRight],0,'post');

% Network input size
Iout = imresize(I,[256 256]);
end