function [] = Plot2DParams (vNorm, is, bnds, chi, ns, nr, MaxIter)

ScreenSize = get(0,'ScreenSize');
FldNames   = fields(is);

% finds the first two non-fixed parameters that were plotted
VarVary  = find(diff(bnds,[],2) >0);
Nvar     = length(VarVary);

Nrow = floor(sqrt(Nvar-1));
Ncol = ceil((Nvar-1)/Nrow);

[~,inds] = sort(chi);

figure;
set(gcf,'Position',[100,100,0.8*ScreenSize(3:4)]);

for ivar = 1:(Nvar-1)
    
    subplot(Nrow,Ncol,ivar);
    PlotVoronoi(vNorm(:,VarVary(ivar+(0:1))), ...
        bnds(VarVary(ivar+(0:1)),:), chi, inds(1:nr));
    
    AddAxes(bnds(VarVary(ivar+(0:1)),:), FldNames(VarVary(ivar+(0:1))));
end



end


function [] = AddAxes (bnds, FldNames)

xlabel([FldNames{1} ', ' num2str(bnds(1,1)) ' to ' num2str(bnds(1,2))]);
ylabel([FldNames{2} ', ' num2str(bnds(2,1)) ' to ' num2str(bnds(2,2))]);

end



