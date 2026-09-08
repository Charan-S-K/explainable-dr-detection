function dataOut = augmentVesselData(data)
% Safe augmentation for DRIVE vessel segmentation

    I = data{1};
    C = data{2};

    % Random horizontal flip
    if rand > 0.5
        I = fliplr(I);
        C = fliplr(C);
    end

    % Random vertical flip
    if rand > 0.5
        I = flipud(I);
        C = flipud(C);
    end

    % IMPORTANT:
    % No rotation for now because rotation may introduce
    % undefined categorical pixels in the vessel mask.

    dataOut = {I,C};
end