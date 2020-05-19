function [] = PlayNeighborhoodSearch (mNorm, mNames, mbnds, L, NbrOpts, iterplt, mTrue, filename)

Nvar = size(mNorm,2);

% finds the first two non-fixed parameters to plot
Vars = find(diff(mbnds,[],2) > 0);

if Nvar == 1, return; end
if isempty(mNames), mNames = repmat({''}, Nvar, 1); end
if isempty(iterplt), iterplt = 1:NbrOpts.Niter; end
if isempty(mTrue), mTrue = nan(Nvar,1); end

Ns = NbrOpts.Ns;
Nr = NbrOpts.Nr;

figHand = figure;
set(gcf, 'color', 'w','defaultlinelinewidth',1);

% open movie file
vidObj = VideoWriter(filename);
vidObj.FrameRate = 2;
vidObj.Quality = 100;

open(vidObj);

for i = iterplt
    
    IndEnd = Ns + (i-1)*floor(Ns/Nr)*Nr;
    inds = NeighborhoodSearch('Lsort',L(1:IndEnd));
    h = voronoi(mNorm(1:IndEnd,Vars(1)), mNorm(1:IndEnd,Vars(2)));
    h(2).Color = 'k';
    hold on;
    
    scatter(mNorm(inds,Vars(1)),mNorm(inds,Vars(2)), 30, 1:IndEnd, 'filled');
    colormap(autumn);
    plot(mTrue(Vars(1)), mTrue(Vars(2)), 'k^', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
    
    set(gca,'XLim', [0,1], 'YLim', [0,1]);
    set(gca,'XTick',[], 'YTick', []);
    
    cb = colorbar;
    cb.Ticks = [];
    cb.Direction = 'reverse';
    title(cb,'Fit to data', 'FontSize', 16);
    text(1.1,1,'Good', 'FontSize', 16, 'VerticalAlignment', 'top');
    text(1.1,0,'Poor', 'FontSize', 16, 'VerticalAlignment', 'bottom');
    
    hold off;
    axis square
    xlabel(mNames{Vars(1)}); ylabel(mNames{Vars(2)});
    
    currFrame = getframe(figHand);
    writeVideo(vidObj,currFrame);
    
end

close(vidObj);
end