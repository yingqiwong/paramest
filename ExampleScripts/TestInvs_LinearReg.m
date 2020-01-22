% This code tests various inversion schemes on a simple linear regression
% model and compares results.
% 
% YQW, Nov 21, 2019.

clear all;
addpath(genpath('../'));
addpath('../../gwmcmc/');
set(0,'defaultlinelinewidth',2);

%% generate true model

Nvars = 2;
% mTrue = rand(Nvars,1);
mTrue = [0.3;0.5];
sigma = 0.1;

N = 1000;
X = rand(N, Nvars-1);
G = [ones(N,1), X];

% synthetic data
data = LinRegFuncs('NoisyData', G, mTrue, sigma);
dTrue = LinRegFuncs('dhatFunc', G, mTrue);

% run a least squares
mlin = (G'*G)\(G'*data);
covm = sigma^2*((G'*G)\eye(2));

%% bounds of model and define functions

mNames = {'a','b'};
mbnds  = [0,1; 0,1];

dhatFunc  = @(model) LinRegFuncs('dhatFunc', G, model);
PriorFunc = @(model) ProbFuncs('PriorFunc',model,mbnds,[],'Uniform');
PrSmpFunc = @(Niter) ProbFuncs('PriorSampFunc', 'Uniform', Niter, mbnds);
LikeFunc  = @(dhat, Linputs) ProbFuncs('LikeFunc',dhat,data,sigma*ones(size(data)),Linputs);
LkMdFunc  = @(model) ProbFuncs('LikeFuncModel', dhatFunc, model, data, sigma*ones(size(data)));


%% run one iteration of the model to ensure functions are correct

xtmp = rand(Nvars,1);
[dhat, ~, Lin] = dhatFunc(xtmp);
pr = PriorFunc(xtmp);
lk = LikeFunc(dhat, Lin);
[lm,dhat] = LkMdFunc(xtmp);

%% make some plots

% sort X for plotting
[Xsort, iX] = sort(X);

figure;
plot(X, data, '^'); hold on;
plot(X(iX), G(iX,:)*mTrue, '-', 'linewidth', 4);
plot(X(iX), G(iX,:)*mlin, '--', 'linewidth', 4);
plot(X(iX), G(iX,:)*xtmp, '+');
xlabel('x_1'); ylabel('y');
legend('Noisy','True','Least squares','random model','location','best'); 
legend boxoff;
drawnow;




%% NOW, test different inversion schemes
%% MCMC
% adjust step size to get reasonable acceptance ratio ~26%

x0    = rand(Nvars,1);
xstep = 0.02*diff(mbnds,[],2); 
Niter = 20000;
BurnIn = 0.1*Niter;

tic;
[x_keep,P_keep,count] = mcmc(dhatFunc,PriorFunc,LikeFunc,x0,xstep,mbnds,Niter);
RunTime(1) = toc;
fprintf('Acceptance ratio = %.2f.\n', count/Niter*100);

xMAP = PlotMCMCAnalytics(x_keep, P_keep, mbnds, count, BurnIn, mNames);
figure; ecornerplot(x_keep,'ks',true,'color',[.6 .35 .3])
ppd_mcmc = CalcPPD(x_keep(BurnIn:end,:), mbnds, 1000);

%% gwmcmc

minit = rand(Nvars, 100);
tic
[mgw, pgw, dhat] =gwmcmc(minit,{PriorFunc LkMdFunc},Niter,'ThinChain',1,'BurnIn',0.3,'OutputData',1);
RunTime(2) = toc;

mgw = mgw(:,:)'; pgw = pgw(:,:)'; dhat = dhat(:,:)';
ppd_gw = CalcPPD(mgw, mbnds, 1000);
figure; ecornerplot(mgw(:,:)','ks',true,'color',[.6 .35 .3])

figure;
plot(X, data, '^'); hold on;
iplt=randi(length(dhat),1,200);
for kk=iplt
    h=plot(X(iX),dhat{kk,2}(iX),'color',[.6 .35 .3].^.3);
    hold on
end
plot(X(iX), dTrue{1}(iX), 'r-', 'linewidth', 4);
xlabel('x_1'); ylabel('y');
legend('Noisy','True','random model','location','best'); 
legend boxoff;
drawnow;

%% catmip

Ncm = 1000;
mtmp = PrSmpFunc(10);
tic
[xcm, LLK, dhcm, rtcm, allx] = catmip(PrSmpFunc, LkMdFunc, mbnds,...
    'Niter', Ncm, 'Nsteps', 5);
RunTime(3) = toc;
ppd_catmip = CalcPPD(xcm, mbnds, 1000);
figure; ecornerplot(xcm,'ks',true,'color',[.6 .35 .3])
PlotTemperingSteps(allx, mbnds, mNames)

figure;
plot(X, data, '^'); hold on;
iplt=randi(length(dhcm),1,200);
for kk=iplt
    h=plot(X(iX),dhcm{kk,1}(iX),'color',[.6 .35 .3].^.3);
    hold on
end
htrue=plot(X(iX), dTrue{1}(iX), 'r-', 'linewidth', 4);
xlabel('x_1'); ylabel('y');
legend([h,htrue],'Random','True','location','best'); 
legend boxoff;
drawnow;

%% Neighborhood algorithm

NbrOpts       = LoadNbrOpts;
NbrOpts.Ns    = 100;
NbrOpts.Nr    = 20;
NbrOpts.Niter = 19;
NbrOpts.plot  = 0;
NbrOpts.Ngibbs= 2000;
NbrOpts.Nchain= 2;
NbrOpts.Nppd  = 1000;

% search
tic;
[mReal, mNorm, ~, L, ~, ~, ~] = NeighborhoodSearch('main',...
    dhatFunc, LikeFunc, mbnds, mNames, NbrOpts, 1);
RunTime(4) = toc;

PlotNAIterations(mNorm, mNames, mbnds, L, NbrOpts, 1:2:20)
AddTrueModelToPlot((mTrue-mbnds(:,1))/diff(mbnds,[],2));
figure; plot(mNorm(:,1), L, '+');

% appraise
[ppd_nbr, mOut, mRealOut, LPxi, mChain] = GibbsSampler('main', mNorm, mbnds, L, NbrOpts);

misfit = - L;
tic
[ppd_nbr2, mOut2] = gibbs_fk (mNorm, misfit, mbnds, NbrOpts);
toc

% ppd_nbr = CalcPPD(xcm, mbnds, 1000);

PlotPosteriorPDF(mRealOut, mbnds, mNames, mTrue);


%% compare pdfs of all the schemes

disp(RunTime);

clear ppd_true;
for mi = 1:length(mNames)
    ppd_true.m(:,mi) = linspace(mbnds(mi,1), mbnds(mi,2), 1000);
    ppd_true.prob(:,mi) = normpdf(ppd_true.m(:,mi), mlin(mi), sqrt(covm(mi,mi)));
end

figure; 
set(gcf,'defaultlinelinewidth', 2, 'Position',[511   667   971   318]);

for mi = 1:Nvars
    subplot(1,Nvars,mi);
    plot(ppd_true.m(:,mi), ppd_true.prob(:,mi), 'k-', 'LineWidth', 4);
    hold on;
    plot(ppd_mcmc.m(:,mi), ppd_mcmc.prob(:,mi));
    plot(ppd_gw.m(:,mi),   ppd_gw.prob(:,mi));
    plot(ppd_catmip.m(:,mi), ppd_catmip.prob(:,mi));
    plot(ppd_nbr.m(:,mi),  ppd_nbr.prob(:,mi));
    plot(ppd_nbr2.m(:,mi),  ppd_nbr2.prob(:,mi));
    plot(mlin(mi)*ones(1,2), ylim, 'k:');
    hold off;
    xlim(mlin(mi) + [-1,1]*10*sqrt(covm(mi,mi)));
    leg = legend('True','MCMC','GWMCMC','CATMIP','NBR','NBR,Fukushima','Least Squares','location','best');
    legend boxoff;
    title(leg, 'Inversion method');
    title(mNames{mi});
end








