function [] = PlotHistograms (m, bnds, ParamNames, nbins, mTrue)

[Niter,Nvar] = size(m);

if nargin < 4, nbins = floor(sqrt(Niter));  end
if nargin < 5, mTrue = nan*ones(1,Nvar);    end

Nrow = floor(sqrt(Nvar));
Ncol = ceil((Nvar)/Nrow);

figure;
for ivar = 1:Nvar
    subplot(Nrow, Ncol, ivar);
    histogram(m(:, ivar), nbins);
    hold on; plot(mTrue(ivar)*ones(1,2), ylim, 'r-'); hold off;
    
    if ~isempty(bnds), xlim(bnds(ivar,:)); end
    if nargin > 2, xlabel(ParamNames{ivar}); end
end

end