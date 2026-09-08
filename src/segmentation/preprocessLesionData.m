function dataOut = preprocessLesionData(data)
% Resize IDRiD image + lesion mask for U-Net training.

targetSize = [256 256];

I = data{1};
C = data{2};

% RGB image
I = imresize(I,targetSize);

% Categorical segmentation mask
C = imresize(C,targetSize,"nearest");

dataOut = {I,C};
end
