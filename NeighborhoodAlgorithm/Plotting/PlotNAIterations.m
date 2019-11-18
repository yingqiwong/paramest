function [] = PlotNAIterations (vNorm, mNames, bnds, L, NbrOpts, iterplt)

Nvar = size(vNorm,2);

if Nvar == 1, return; end
if isempty(mNames), mNames = repmat({''}, Nvar, 1); end
if isempty(iterplt), iterplt = 1:NbrOpts.Niter; end

Ns = NbrOpts.Ns;
Nr = NbrOpts.Nr;

ScreenSize = get(0,'ScreenSize');

[Nrow, Ncol] = GetSubplotRowCol(length(iterplt));

figure;
set(gcf,'Position',[100,100,0.8*ScreenSize(3:4)]);

subplot(Nrow,Ncol,1);
inds = NeighborhoodSearch('Lsort',L(1:Ns));
PlotVoronoi(vNorm(1:Ns,:), bnds, L(1:Ns), inds(1:Nr));
    
for i = 2:length(iterplt)
    IndEnd = Ns + (iterplt(i)-1)*floor(Ns/Nr)*Nr;
   
    subplot(Nrow,Ncol,i);
    inds = NeighborhoodSearch('Lsort',L(1:IndEnd));
    PlotVoronoi(vNorm(1:IndEnd,:), bnds, L(1:IndEnd), inds(1:Nr));
    
    AddAxes(bnds, mNames);
end

end


function [] = AddAxes (bnds, FldNames)

xlabel([FldNames{1} ', ' num2str(bnds(1,1)) ' to ' num2str(bnds(1,2))]);
ylabel([FldNames{2} ', ' num2str(bnds(2,1)) ' to ' num2str(bnds(2,2))]);

end