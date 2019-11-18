function [xMAP] = PlotMCMCAnalytics (x, P, mbnds, count, BurnIn, VarNames)

[Niter, Nvar] = size(x);

Nrow = floor(sqrt(Nvar));
Ncol = ceil(Nvar/Nrow);

[~, xMAPind] = max(P);
xMAP = x(xMAPind,:);

figure;
for mi = 1:Nvar
    subplot(Nrow,Ncol,mi);
    histogram(x(BurnIn:end,mi), 40);
    hold on; 
    plot(mbnds(mi,1)*ones(1,2), ylim, 'r-');
    plot(mbnds(mi,2)*ones(1,2), ylim, 'r-');
    plot(xMAP(mi)*ones(1,2), ylim, 'r:');
    hold off;
    title(VarNames{mi});
end
suptitle('Posterior PDFs');

figure; semilogy(P); 
title(['Acceptance ratio = ' num2str(100*count/Niter,4)]);

end