function [xMAP] = PlotMCMCAnalytics (x, P, mbnds, count, BurnIn, VarNames)

[Niter, Nvar] = size(x);

Nrow = floor(sqrt(Nvar));
Ncol = ceil(Nvar/Nrow);

% find the maximum a posteriori model
[~, xMAPind] = max(P);
xMAP = x(xMAPind,:);

VarVary = diff(mbnds,[],2)>0;

figure;

% plot distributions of the model parameters
for mi = 1:Nvar
    subplot(Nrow,Ncol,mi);
    
    % plot the distribution 
    histogram(x(BurnIn:end,mi), 40);   hold on; 
    
    % bounds
    plot(mbnds(mi,1)*ones(1,2), ylim, 'r-');
    plot(mbnds(mi,2)*ones(1,2), ylim, 'r-');
    
    % maximum a posteriori model
    plot(xMAP(mi)*ones(1,2), ylim, 'r:');
    hold off;
    
    if VarVary(mi); xlim(mbnds(mi,:)); end
    title(VarNames{mi});
end
sgtitle('Posterior PDFs');

% plot the chain to see if well mixed
figure; semilogy(P); 
title(['Acceptance ratio = ' num2str(100*count/Niter,4)]);

end