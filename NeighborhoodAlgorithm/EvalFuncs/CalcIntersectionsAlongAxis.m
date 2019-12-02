function [xji, xcell] = CalcIntersectionsAlongAxis (v, xA, ivar)
% calculates the intersection points of other Voronoi cells along the
% ivar-th axis, starting with the xA point

% reset xA(ivar) to the lowest limit and start searching from there
xAsearch        = xA;
xAsearch(ivar)  = 0;

% initialize maximum size of outputs
Npts    = size(v,1);
xji     = zeros(Npts, 2);
xcell   = zeros(Npts, 1);

% find nearest cell node by minimizing distance between v and xA
distances       = sqrt(sum((v - xAsearch).^2, 2)); 
[~, xcell(1)]   = min(distances);
xAsearch(ivar)  = eps; 
% add epsilon to perturb position from lower bound

count = 1;
upper = 0;

% loop over all neighboring cells to find intersections until the upper
% bound is reached. 
while upper < 1
    
    % calculate the upper and lower limits that intersect the current axis
    [lower, upper, lInd, uInd] = CalcLimits(v, xAsearch, ivar);
    
    xji(count, :) = [lower, upper];
    
    if isempty(uInd), break; end

    % record cell number
    xcell(count+1) = uInd;
    
    % perturb position to just a bit to the right of the upper limit of the
    % previous cell
    xAsearch(ivar) = upper + eps;

    count = count+1;
end

% throw away excess elements of the vector (these should still be zeros)
xji(count+1:end,:) = [];
xcell(count+1:end) = [];

end