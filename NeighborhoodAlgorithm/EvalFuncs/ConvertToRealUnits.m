function [mReal] = ConvertToRealUnits (mNorm, bnds)

bnds  = bnds';
mReal = bnds(1,:) + (bnds(2,:) - bnds(1,:)).*mNorm;


end