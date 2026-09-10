function loss = softExudateV2Loss(Y,T)
% SeeBeyond - Soft Exudate V2 loss
%
% Hybrid:
%   60% weighted cross entropy
%   40% lesion-only Dice loss

%% Weighted cross entropy

weights = dlarray([1; 2.5],"C");

ceLoss = crossentropy( ...
    Y,T, ...
    weights, ...
    NormalizationFactor="all-elements");

%% Soft-exudate class only
% Channel 1 = background
% Channel 2 = soft exudate

P = Y(:,:,2,:);
G = T(:,:,2,:);

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

%% Apply Dice only to patches containing lesion

hasLesion = gtSum > 0;

diceLoss = ...
    sum((1-diceScore).*hasLesion) ./ ...
    (sum(hasLesion)+epsilon);

%% Final loss

loss = ...
    0.60 .* ceLoss + ...
    0.40 .* diceLoss;

end