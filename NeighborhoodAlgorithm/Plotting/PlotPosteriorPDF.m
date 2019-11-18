function [] = PlotPosteriorPDF (m, bnds, ParamNames, mTrue)

[Niter,Nvar] = size(m);

if nargin < 4, mTrue = nan*ones(1,Nvar);    end

Nrow = floor(sqrt(Nvar));
Ncol = ceil((Nvar)/Nrow);

figure;
for ivar = 1:Nvar
    subplot(Nrow, Ncol, ivar);
    
    [ppd, mi] = ksdensity(m(:,ivar));
    plot(mi, ppd, 'linewidth', 2);
    hold on; plot(mTrue(ivar)*ones(1,2), ylim, 'r-'); hold off;
    
    if ~isempty(bnds), xlim(bnds(ivar,:)); end
    if nargin > 2, xlabel(ParamNames{ivar}); end
    
end


end