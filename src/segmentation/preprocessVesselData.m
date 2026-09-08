function dataOut = preprocessVesselData(data)
% preprocessVesselData
% Resize DRIVE image and vessel mask to 256x256.

targetSize = [256 256];

I = data{1};
C = data{2};

% Resize RGB retinal image
I = imresize(I,targetSize);

% Resize categorical vessel mask
% Keep label boundaries intact
C = imresize(C,targetSize,"nearest");

dataOut = {I,C};
end