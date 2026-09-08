function dataOut = augmentLesionData(data)

I = data{1};
C = data{2};

% Horizontal flip only
if rand > 0.5
    I = fliplr(I);
    C = fliplr(C);
end

dataOut = {I,C};
end