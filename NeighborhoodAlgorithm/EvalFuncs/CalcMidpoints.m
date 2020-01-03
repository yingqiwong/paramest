function [xji] = CalcMidpoints (v, xA, ivar)
% calculates the midpoints between point of interest and all other cell
% nodes projected onto the ivar-th axis 
% (equation 19 of Sambridge 1999 I)
% ivar = variable (axis) index

[Npts, Nvars] = size(v);

% find nearest cell node by minimizing distance between v and xA
distances = sqrt(sum((v - xA).^2, 2)); 
[~, k]    = min(distances);

% point of interest
xAperp = xA;
xAperp(ivar) = v(k,ivar);

dk2 = norm(v(k,:) - xAperp).^2;

xji = zeros(Npts, 1);
for ix = 1:Npts
    if ix == k, continue; end
    
    vperp = xA;
    vperp(ivar) = v(ix,ivar);
    
    dj2 = norm(v(ix,:) - vperp).^2;
    xji(ix) = 0.5*(v(k,ivar)+v(ix,ivar) + (dk2 - dj2)/(v(k,ivar)-v(ix,ivar)));
    
end

end