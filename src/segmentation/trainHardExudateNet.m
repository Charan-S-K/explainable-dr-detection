function dataOut = preprocessHardExudateData(data)
% Preserve IDRiD aspect ratio and resize to 512x512.

I = data{1};
C = data{2};

targetH = 512;
targetW = 512;

% Convert lesion labels to binary mask
M = C == "hard_exudate";

[h,w,~] = size(I);

% Preserve aspect ratio
scale = min(targetH/h,targetW/w);

newH = round(h*scale);
newW = round(w*scale);

I = imresize(I,[newH newW]);
M = imresize(M,[newH newW],"nearest");

% Calculate padding
padTop    = floor((targetH-newH)/2);
padBottom = targetH-newH-padTop;

padLeft   = floor((targetW-newW)/2);
padRight  = targetW-newW-padLeft;

% Pad image
I = padarray(I,[padTop padLeft],0,"pre");
I = padarray(I,[padBottom padRight],0,"post");

% Pad mask
M = padarray(M,[padTop padLeft],false,"pre");
M = padarray(M,[padBottom padRight],false,"post");

% Back to categorical labels
C = categorical(double(M), ...
    [0 1], ...
    ["background","hard_exudate"]);

dataOut = {I,C};

end