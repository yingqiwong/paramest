function [lower, cell, upper, lInd, uInd] = CalcLimits (v, xA, ivar)
% calculates the lower and upper limits of sampling within one cell 
% each variable has max range [0,1]
% (equations 20, 21 in Sambridge paper)

[xji, cell] = CalcMidpoints(v, xA, ivar);

LowerInds = find(xji<=xA(ivar));
UpperInds = find(xji>=xA(ivar));

[lower, lIndTmp] = max([xji(LowerInds); 0]);
[upper, uIndTmp] = min([xji(UpperInds); 1]);

lInd = []; uInd = [];

if lower ~= 0
    lInd = LowerInds(lIndTmp);
end

if upper ~= 1
    uInd = UpperInds(uIndTmp);
end

end