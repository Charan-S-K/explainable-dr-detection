function loss = tverskyLossLesion(Y,T)
% Tversky loss for highly imbalanced lesion segmentation.
% Higher beta penalizes missed lesion pixels more strongly.

alpha = 0.3;   % false-positive penalty
beta  = 0.7;   % false-negative penalty

Ynot = 1 - Y;
Tnot = 1 - T;

TP = sum(sum(Y .* T,1),2);
FP = sum(sum(Y .* Tnot,1),2);
FN = sum(sum(Ynot .* T,1),2);

epsilon = 1e-8;

tversky = ...
    (TP + epsilon) ./ ...
    (TP + alpha*FP + beta*FN + epsilon);

lossPerClass = 1 - tversky;

lossPerImage = sum(lossPerClass,3);

N = size(Y,4);
loss = sum(lossPerImage) / N;
end