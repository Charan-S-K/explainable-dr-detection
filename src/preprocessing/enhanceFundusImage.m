function enhanced = enhanceFundusImage(I)
% enhanceFundusImage
% Mild denoising + CLAHE enhancement for retinal fundus images.

% Ensure RGB image
if size(I,3) ~= 3
    I = repmat(I,[1 1 3]);
end

I = im2uint8(I);

% Mild Gaussian denoising
denoised = imgaussfilt(I,0.7);

% Convert RGB -> LAB
lab = rgb2lab(denoised);

% Luminance channel range is approximately 0-100
L = lab(:,:,1) / 100;

% CLAHE
L = adapthisteq(L, ...
    'NumTiles',[8 8], ...
    'ClipLimit',0.01);

% Restore LAB luminance range
lab(:,:,1) = L * 100;

% LAB -> RGB
enhanced = lab2rgb(lab,'OutputType','uint8');

end