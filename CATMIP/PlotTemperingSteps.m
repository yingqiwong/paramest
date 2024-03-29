function [] = PlotTemperingSteps (allmodels, mbnds, mNames)
% PlotTemperingSteps (allmodels, mbnds, mNames)
%
% plots distribution for each tempering step from catmip output allmodels
%
% YQW Dec 27, 2019

[Niter, Nvar, Ntemp] = size(allmodels);
if nargin<3, mNames = repmat({''}, Nvar, 1); end

figure;
set(gcf,'defaultlinelinewidth', 1, 'defaultaxescolororder', parula(Ntemp+1));
[Nrow, Ncol] = GetSubplotRowCol(Nvar);

for vi = 1:Nvar
    subplot(Nrow, Ncol, vi);
    
    % specify bins for calculating histograms to be constant for all the
    % tempering steps
    vbins = linspace(mbnds(vi,1), mbnds(vi,2), floor(Niter/20));
    dx    = vbins(2) - vbins(1);

    % plot histograms for tempering steps
    for ti = 1:Ntemp
        n = histcounts(allmodels(:,vi,ti), vbins);        
        plot(vbins(1:end-1)+0.5*dx, 1/Niter/dx*n); hold on;
    end
    
    hold off;
    title(mNames{vi});
end

end