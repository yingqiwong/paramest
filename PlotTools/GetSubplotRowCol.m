function [Nrow, Ncol] = GetSubplotRowCol (N, Nrow, Ncol)

if nargin > 1
    if isempty(Nrow)
        Nrow = N/Ncol;
    elseif isempty(Ncol)
        Ncol = N/Nrow;
    end
else
    Nrow = floor(sqrt(N));
    Ncol = ceil(sqrt(N));
end

end