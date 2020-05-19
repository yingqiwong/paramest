% This code tests various inversion schemes on a simple linear regression
% model and compares results.
% 
% YQW, Nov 21, 2019.

clear all;
addpath(genpath('../'));
addpath('../../gwmcmc/');
set(0,'defaultlinelinewidth',2, 'defaultaxesfontsize', 16);

%% generate true model

Nvars = 2;
% mTrue = rand(Nvars,1);
mTrue = [4;5];
sigma = 1e-3;

N = 101;
r = 20*rand(N,1);
nu = 0.25;

% synthetic data
data = MogiFuncs('NoisyData', mTrue, r, nu, sigma);

%% bounds of model and define functions

mNames = {'dV','D'};
mbnds  = [0,10;0,10];

dhatFunc  = @(model) MogiFuncs('dhatFunc', model,r,nu);
PriorFunc = @(model) ProbFuncs('PriorFunc',model,mbnds,[],'Uniform');
PrSmpFunc = @(Niter) ProbFuncs('PriorSampFunc', 'Uniform', Niter, mbnds);
LikeFunc  = @(dhat, Linputs) ProbFuncs('LikeFunc',dhat,data,sigma*ones(size(data)),Linputs);
LkMdFunc  = @(model) ProbFuncs('LikeFuncModel', dhatFunc, model, data, sigma*ones(size(data)));


%% run one iteration of the model to ensure functions are correct

xtmp = mbnds(:,1) + diff(mbnds,[],2).*rand(2,1);
[dhat, ~, Lin] = dhatFunc(xtmp);
[pr,junk] = PriorFunc(xtmp);
lk = LikeFunc(dhat, Lin);
[lm,dhat2] = LkMdFunc(xtmp);

%% make some plots

dTrue = dhatFunc(mTrue);
dxTmp = dhatFunc(xtmp);

% sort r for plotting
[Rsort, ir] = sort(r);

figure;
plot(r, data, '^'); hold on;
plot(r(ir), dTrue{1}(ir), '-', 'linewidth', 4);
plot(r(ir), dxTmp{1}(ir), '+');
xlabel('x_1'); ylabel('y');
legend('Noisy','True','random model','location','best'); 
legend boxoff;
drawnow;




%% NOW, test different inversion schemes
% establish some common traits so that we can compare the methods
Niter = 1000000;

%% MCMC
% adjust step size to get reasonable acceptance ratio ~26%

x0    = mbnds(:,1) + diff(mbnds,[],2).*rand(Nvars,1);
xstep = 0.02*diff(mbnds,[],2); 
% xstep = rand(1)*diff(mbnds,[],2); % random step size
BurnIn = 0.1*Niter;

tic;
[m_mcmc,P_mcmc,count] = mcmc(dhatFunc,PriorFunc,LikeFunc,x0,xstep,mbnds,Niter);
RunTime(1) = toc;
fprintf('Acceptance ratio = %.2f.\n', count/Niter*100);

xMAP = PlotMCMCAnalytics(m_mcmc, P_mcmc, mbnds, count, BurnIn, mNames);
[ppd_mcmc.m, ppd_mcmc.prob] = CalcPDF(mbnds, m_mcmc(BurnIn:end,:), Niter/50);

%% gwmcmc

minit = mbnds(:,1) + diff(mbnds,[],2).*rand(Nvars,100);

tic
[m_gw, p_gw, dhat] =gwmcmc(minit,{PriorFunc LkMdFunc},Niter,'ThinChain',1, ...
    'BurnIn',0.2,'OutputData',1,'FileName','','ProgressBar',false);
RunTime(2) = toc;

m_gw = m_gw(:,:)'; p_gw = p_gw(:,:)'; dhat = dhat(:,:)';
[ppd_gw.m, ppd_gw.prob] = CalcPDF(mbnds, m_gw, Niter/50);
% figure; ecornerplot(mgw(:,:)','ks',true,'color',[.6 .35 .3])

%% catmip

Ncm = 400;
mtmp = PrSmpFunc(10);
tic
[m_catmip, p_catmip, dhcm, rtcm, m_catmip_all] = catmip(PrSmpFunc, LkMdFunc, mbnds,...
    'Niter', Ncm, 'Nsteps', 5);
RunTime(3) = toc;
[ppd_catmip.m, ppd_catmip.prob] = CalcPDF(mbnds, m_catmip, Niter/50);
% figure; ecornerplot(xcm,'ks',true,'color',[.6 .35 .3])
PlotTemperingSteps(m_catmip_all, mbnds, mNames)

%% Neighborhood algorithm

NbrOpts       = LoadNbrOpts;
NbrOpts.Ns    = 8;
NbrOpts.Nr    = 8;
NbrOpts.Niter = 10; %floor((Niter-NbrOpts.Ns)/NbrOpts.Ns);
NbrOpts.plot  = 0;
NbrOpts.Ngibbs= 500;
NbrOpts.Nchain= 2;
NbrOpts.Nppd  = 250;

% search
tic;
[mReal, mNorm, ~, L, ~, ~, ~] = NeighborhoodSearch('main',...
    dhatFunc, LikeFunc, mbnds, mNames, [], NbrOpts, 1);
RunTime(4) = toc;

PlotNAIterations(mNorm, mNames, mbnds, L, NbrOpts, 1:NbrOpts.Niter)
mTrueNorm = (mTrue-mbnds(:,1))/diff(mbnds,[],2);
AddTrueModelToPlot(mTrueNorm);
PlayNeighborhoodSearch(mNorm, {'Parameter 1', 'Parameter 2'}, mbnds, L, NbrOpts, 1:NbrOpts.Niter, mTrueNorm, 'tmp')

% appraise
[ppd_nbr, mOut, mRealOut, LPxi, mChain] = GibbsSampler('main', mNorm, mbnds, L, NbrOpts);

misfit = - L;
[ppd_nbr2, mOut2] = gibbs_fk (mNorm, misfit, mbnds, NbrOpts);

%% compare pdfs of all the schemes

disp(RunTime);

figure; 
set(gcf,'defaultlinelinewidth', 2, 'Position',[511   667   971   318]);

for mi = 1:Nvars
    subplot(1,Nvars,mi);
    plot(ppd_mcmc.m(:,mi), ppd_mcmc.prob(:,mi));
    hold on;
    plot(ppd_gw.m(:,mi),   ppd_gw.prob(:,mi));
    plot(ppd_catmip.m(:,mi), ppd_catmip.prob(:,mi));
    plot(ppd_nbr.m(:,mi),  ppd_nbr.prob(:,mi));
%     plot(ppd_nbr2.m(:,mi),  ppd_nbr2.prob(:,mi));
    plot(mTrue(mi)*ones(1,2), ylim, 'k:');
    hold off;
    xlim(mTrue(mi) + [-1,1]*10*std(m_mcmc(:,mi)));
    leg = legend('MCMC','GWMCMC','CATMIP','NBR','NBR,Fukushima','location','best'); 
    legend boxoff;
    title(leg, 'Inversion method');
    title(mNames{mi});
end

%% plot model outputs

figure;
iplt=randi(length(dhcm),1,40);
for kk=iplt
    h=plot(r(ir),dhcm{kk}(ir),'color',[.6 .35 .3].^.3);
    hold on
end
hdat  = errorbar(r, data, sigma*ones(size(data)), '^','color',lines(1)); hold on;
htrue = plot(r(ir), dTrue{1}(ir), 'r-', 'linewidth', 4);
xlabel('x_1'); ylabel('y');
legend([hdat;htrue],{'Noisy';'True'},'location','best'); 
legend boxoff;
drawnow;



%% junk

figure;
subplot(211);
histogram(m_mcmc(BurnIn:end,1), 200, 'EdgeColor', 'none');
xlim([3.5,4.5]);
set(gca,'box','off','xtick', [], 'ytick', []);
xlabel('Parameter 1');

subplot(212);
histogram(m_mcmc(BurnIn:end,2), 200, 'EdgeColor', 'none');
xlim([4,5.5]);
set(gca,'box','off','xtick', [], 'ytick', []);
xlabel('Parameter 2');
