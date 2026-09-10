function loss = haemorrhageV4Loss(Y,T)
% SeeBeyond - Haemorrhage V4 loss
%
% Stable hybrid:
%   60% weighted cross entropy
%   40% lesion-only Dice loss
%
% Dice is calculated ONLY for the haemorrhage class.
% Empty negative patches do not dominate the Dice term.

%% Weighted CE

weights = dlarray([1; 3],"C");

ceLoss = crossentropy( ...
    Y,T, ...
    weights, ...
    NormalizationFactor="all-elements");

%% Haemorrhage channel only

P = Y(:,:,2,:);
G = T(:,:,2,:);

%% Sum per image

intersection = ...
    sum(sum(P .* G,1),2);

predSum = ...
    sum(sum(P,1),2);

gtSum = ...
    sum(sum(G,1),2);

epsilon = 1e-6;

diceScore = ...
    (2 .* intersection + epsilon) ./ ...
    (predSum + gtSum + epsilon);

%% Only apply Dice to images containing haemorrhage

hasLesion = gtSum > 0;

numerator = ...
    sum((1 - diceScore) .* hasLesion);

denominator = ...
    sum(hasLesion) + epsilon;

diceLoss = numerator ./ denominator;

%% Final hybrid loss

loss = ...
    0.60 .* ceLoss + ...
    0.40 .* diceLoss;

end