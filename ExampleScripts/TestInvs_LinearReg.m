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
covm = sigma^2*((G'*G)\eye(Nvars));

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

%% plot the synthetic data

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
% establish some common traits so that we can compare the methods
Niter = 10000;


%% MCMC
% adjust step size to get reasonable acceptance ratio ~26%

x0    = rand(Nvars,1);
xstep = 0.02*diff(mbnds,[],2); % good one
% xstep = rand(1)*diff(mbnds,[],2); % random step size
BurnIn = 0.1*Niter;

tic;
[m_mcmc,P_mcmc,count] = mcmc(dhatFunc,PriorFunc,LikeFunc,x0,xstep,mbnds,Niter);
RunTime(1) = toc;
fprintf('Acceptance ratio = %.2f.\n', count/Niter*100);

xMAP = PlotMCMCAnalytics(m_mcmc, P_mcmc, mbnds, count, BurnIn, mNames);
[ppd_mcmc.m, ppd_mcmc.prob] = CalcPDF(mbnds, m_mcmc(BurnIn:end,:), Niter/50);

%% gwmcmc

minit = rand(Nvars, 200);
tic
[m_gw, p_gw, dhat] = gwmcmc(minit,{PriorFunc LkMdFunc},Niter,'ThinChain',1,'BurnIn',0.2,'OutputData',1);
RunTime(2) = toc;

m_gw = m_gw(:,:)'; p_gw = p_gw(:,:)'; dhat = dhat(:,:)';
[ppd_gw.m, ppd_gw.prob] = CalcPDF(mbnds, m_gw, Niter/50);
% figure; ecornerplot(m_gw(:,:)','ks',true,'color',[.6 .35 .3])

%% catmip

Ncm = 100;
mtmp = PrSmpFunc(10);
tic
[m_catmip, p_catmip, dhcm, rtcm, m_catmip_all] = catmip(PrSmpFunc, LkMdFunc, mbnds,...
    'Niter', Ncm, 'Nsteps', 5);
RunTime(3) = toc;
[ppd_catmip.m, ppd_catmip.prob] = CalcPDF(mbnds, m_catmip, Niter/50);
PlotTemperingSteps(m_catmip_all, mbnds, mNames)

%% Neighborhood algorithm

NbrOpts       = LoadNbrOpts;
NbrOpts.Ns    = 200;
NbrOpts.Nr    = 20;
NbrOpts.Niter = 20; %floor((Niter-NbrOpts.Ns)/NbrOpts.Ns);
NbrOpts.plot  = 0;
NbrOpts.Ngibbs= 1000;
NbrOpts.Nchain= 2;
NbrOpts.Nppd  = 400;

% search
tic;
[mReal, mNorm, ~, L, ~, ~, ~] = NeighborhoodSearch('main',...
    dhatFunc, LikeFunc, mbnds, mNames, [], NbrOpts, 1);
RunTime(4) = toc;

PlotNAIterations(mNorm, mNames, mbnds, L, NbrOpts, 1:2:20)
AddTrueModelToPlot((mTrue-mbnds(:,1))/diff(mbnds,[],2));
% figure; plot(mNorm(:,1), L, '+');

% appraise
[ppd_nbr, mOut, mRealOut, LPxi, mChain] = GibbsSampler('main', mNorm, mbnds, L, NbrOpts);

% misfit = - L;
% tic
% [ppd_nbr2, mOut2] = gibbs_fk (mNorm, misfit, mbnds, NbrOpts);
% toc

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
%     plot(ppd_nbr2.m(:,mi),  ppd_nbr2.prob(:,mi));
    plot(mlin(mi)*ones(1,2), ylim, 'k:');
    hold off;
    xlim(mlin(mi) + [-1,1]*10*sqrt(covm(mi,mi)));
    leg = legend('True','MCMC','GWMCMC','CATMIP','NBR','NBR,Fukushima','Least Squares','location','best');
    legend boxoff;
    title(leg, 'Inversion method');
    title(mNames{mi});
end

%% plot output models
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
