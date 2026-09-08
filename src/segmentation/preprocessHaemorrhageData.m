function dataOut = preprocessHaemorrhageData(data)
% Prepare retinal image + haemorrhage mask for 512x512 U-Net.

I = data{1};
C = data{2};

targetH = 512;
targetW = 512;

M = C == "haemorrhage";

[h,w,~] = size(I);

scale = min(targetH/h,targetW/w);

newH = round(h*scale);
newW = round(w*scale);

I = imresize(I,[newH newW]);
M = imresize(M,[newH newW],"nearest");

padTop    = floor((targetH-newH)/2);
padBottom = targetH-newH-padTop;

padLeft   = floor((targetW-newW)/2);
padRight  = targetW-newW-padLeft;

I = padarray(I,[padTop padLeft],0,"pre");
I = padarray(I,[padBottom padRight],0,"post");

M = padarray(M,[padTop padLeft],false,"pre");
M = padarray(M,[padBottom padRight],false,"post");

C = categorical(double(M), ...
    [0 1], ...
    ["background","haemorrhage"]);

dataOut = {I,C};

end