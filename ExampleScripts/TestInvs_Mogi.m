% This code tests various inversion schemes on a simple linear regression
% model and compares results.
%
% YQW, Nov 21, 2019.

clear all; close all;
addpath(genpath('../'));
set(0,'defaultlinelinewidth',2);
rng(16);

%% generate true model

Nvars = 2;          % number of model parameters
mTrue = [2;3];      % true model parameters
sigma = 2e-3;       % data error

N  = 401;           % number of data points
r  = 20*rand(N,1);  % generate radius at which deformation is calculated
nu = 0.25;          % poisson's ratio

% synthetic data
data  = MogiFuncs('NoisyData', mTrue, r, nu, sigma);

%% bounds of model and define functions

mNames = {'dV','D'};
mbnds  = [0,5; 0,5];

% function to calculate forward model
% m --> dhat
dhatFunc  = @(model) MogiFuncs('dhatFunc', model,r,nu);

% function to calculate prior probability given a set of model param values
% m --> prior prob
PriorFunc = @(model) ProbFuncs('PriorFunc',model,mbnds,[],'Uniform');

% function to sample from the prior probability distributionn
% [] --> set of model parameter values [Niter x Nm]
PrSmpFunc = @(Niter) ProbFuncs('PriorSampFunc', 'Uniform', Niter, mbnds);

% function to calculate likelihood of dhat
% dhat --> likelihood
LikeFunc  = @(dhat) ProbFuncs('LikeFunc',dhat,data,sigma*ones(size(data)));

% function to calculate likelihood from model parameter values
% model --> dhat --> likelihood
LkMdFunc  = @(model) ProbFuncs('LikeFuncModel', dhatFunc, model, data, sigma*ones(size(data)));


%% run one iteration of the model to ensure functions are correct

mtmp = mbnds(:,1) + diff(mbnds,[],2).*rand(2,1);
[dhat, flag] = dhatFunc(mtmp);
[pr,junk] = PriorFunc(mtmp);
mtmpsmp = PrSmpFunc(10);
lk = LikeFunc(dhat);
[lm,dhat2] = LkMdFunc(mtmp);

%% plot synthetic data

dTrue = dhatFunc(mTrue);
dxTmp = dhatFunc(mtmp);

% sort r for plotting
[Rsort, ir] = sort(r);

figure;
plot(r    , data     , '^'); hold on;
plot(r(ir), dTrue(ir), '-', 'linewidth', 4);
plot(r(ir), dxTmp(ir), '-', 'linewidth', 4);
xlabel('distance from center');
ylabel('deformation');
legend('Noisy','True','random model','location','best');
legend boxoff;
drawnow;

%% NOW, test different inversion schemes
% establish some common traits so that we can compare the methods

Niter = 100000;
Nbins = min(500,Niter/20);

%% MCMC
% adjust step size to get reasonable acceptance ratio ~26%

x0    = mbnds(:,1) + diff(mbnds,[],2).*rand(Nvars,1);
xstep = 0.015*diff(mbnds,[],2);
BurnIn = 0.1*Niter;

tic;
[m_mcmc,P_mcmc,count] = mcmc(dhatFunc,PriorFunc,LikeFunc,x0,xstep,mbnds,Niter);
RunTime(1) = toc;

xMAP = PlotMCMCAnalytics(m_mcmc, P_mcmc, [], mbnds, count, BurnIn, mNames);
[ppd_mcmc.m, ppd_mcmc.prob] = CalcPDF(mbnds, m_mcmc(BurnIn:end,:), Nbins);

%% gwmcmc


if isfolder('../../gwmcmc')
    addpath('../../gwmcmc/');
    
    minit = mbnds(:,1) + diff(mbnds,[],2).*rand(Nvars,100);
    tic
    [m_gw, p_gw, dhat] =gwmcmc(minit,{PriorFunc LkMdFunc},Niter,'ThinChain',1,'BurnIn',0.2);
    RunTime(2) = toc;
    
    m_gw = m_gw(:,:)'; p_gw = p_gw(:,:)'; dhat = dhat(:,:)';
    [ppd_gwmc.m, ppd_gwmc.prob] = CalcPDF(mbnds, m_gw, Nbins);
end

%% catmip

Nsteps = 20;    % number of tempering steps
tic
[m_catmip, p_catmip, dhcm, rtcm, m_catmip_all] = catmip(PrSmpFunc, LkMdFunc, mbnds,...
    'Niter', Niter/Nsteps, 'Nsteps', Nsteps);
RunTime(3) = toc;
[ppd_catm.m, ppd_catm.prob] = CalcPDF(mbnds, m_catmip, Nbins);
PlotTemperingSteps(m_catmip_all, mbnds, mNames)

%% Neighborhood algorithm

NbrOpts       = LoadNbrOpts;
NbrOpts.Ns    = 50;
NbrOpts.Nr    = 20;
NbrOpts.Niter = 20;
NbrOpts.plot  = 0;
NbrOpts.Ngibbs= 1000;
NbrOpts.Nchain= 2;

%  search
nbt = tic;
[mReal, mNorm, ~, L, ~, ~, ~] = NeighborhoodSearch('main',...
    dhatFunc, LikeFunc, mbnds, mNames, [], NbrOpts, 1);

% appraise
[~, mOut, mRealOut] = GibbsSampler('main', mNorm, mbnds, L, NbrOpts);
[ppd_nbrh.m, ppd_nbrh.prob] = CalcPDF(mbnds, mRealOut, Nbins);
RunTime(4) = toc(nbt);

% plot
PlotNAIterations(mNorm, mNames, mbnds, L, NbrOpts, 1:2:20)
AddTrueModelToPlot((mTrue-mbnds(:,1))/diff(mbnds,[],2));

%% compare pdfs of all the schemes

disp(RunTime);

figure;
set(gcf,'defaultlinelinewidth', 2, 'Position', [500,600,1000,500]);

for mi = 1:Nvars
    subplot(1,Nvars,mi);
    plot(mTrue(mi)); hold on;
    plot(ppd_mcmc.m(:,mi), ppd_mcmc.prob(:,mi)); 
    plot(ppd_gwmc.m(:,mi), ppd_gwmc.prob(:,mi));
    plot(ppd_catm.m(:,mi), ppd_catm.prob(:,mi));
    plot(ppd_nbrh.m(:,mi), ppd_nbrh.prob(:,mi));
    plot(mTrue(mi)*ones(1,2), ylim, 'k-');
    hold off;
    
    leg = legend('','MCMC','GWMCMC','CATMIP','NBR','True','location','southoutside');
    legend boxoff
    title(leg, 'Inversion method');
    
    title(mNames{mi});
end
