function [ppd] = CalcPPD (m, mBnds, Nppd)

Nvar     = size(m,2);

ppd.m    = zeros(Nppd, Nvar);
ppd.prob = zeros(Nppd, Nvar);

for ivar = 1:Nvar
    ppd.m(:,ivar)     = linspace(mBnds(ivar,1), mBnds(ivar,2),Nppd)';
    ppd.prob (:,ivar) = ksdensity(m(:,ivar),ppd.m(:,ivar));
end

end
