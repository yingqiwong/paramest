function [] = PlotVoronoi (v, bnds, L, EvalInds)
% figure; PlotVoronoi(v,bnds,chi,[]);

if size(v,2) == 1, return; end

inds = NeighborhoodSearch('Lsort',L);

% finds the first two non-fixed parameters to plot
BndsDiff = diff(bnds,[],2);
VarVary  = find(BndsDiff >0);

voronoi(v(:,VarVary(1)),v(:,VarVary(2))); hold on;
scatter(v(inds,VarVary(1)),v(inds,VarVary(2)), 40, 1:length(L), 'filled'); 
%colorbar;
plot(v(EvalInds,VarVary(1)), v(EvalInds,VarVary(2)), 'rx','linewidth',2);
axis square

title([num2str(size(v,1)) ' models']);
drawnow;

end