function [xji, k] = CalcMidpoints (v, xA, ivar)
% calculates the midpoints between point of interest and all other cell
% nodes projected onto the ivar-th axis 
% (equation 19 of Sambridge 1999 I)
% ivar = variable (axis) index

[Npts, Nvars] = size(v);

% find nearest cell node by minimizing distance between v and xA
ds     = sum((v - xA).^2, 2); 
[~, k] = min(ds);
vk     = v(k,:);

dk2 = norm((vk - xA)).^2 - (vk(ivar)-xA(ivar)).^2;

xji = nan(Npts, 1);

for ix = 1:Npts
    
    if ix == k, continue; end

    dj2 = norm((v(ix,:) - xA)).^2 - (v(ix,ivar) - xA(ivar)).^2;

    xji(ix) = 0.5*(vk(ivar)+v(ix,ivar) + (dk2 - dj2)/(vk(ivar)-v(ix,ivar)));
    
end

end