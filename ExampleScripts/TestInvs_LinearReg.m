% This code tests various inversion schemes on a simple linear regression
% model and compares results.
%
% YQW, Nov 21, 2019.

clear all; close all;
addpath(genpath('../'));
set(0,'defaultlinelinewidth',2);
rng(16);    % for reproducibility

%% generate true model

Nvars = 2;              % number of model parameters
mTrue = [0.3;0.5];      % true model parameters 
sigma = 0.1;            % data error

N = 1000;               % number of data points
x = rand(N, 1);         % generate dependent variable x
G = [ones(N,1), x];     % create kernel matrix for least squares inversion

% synthetic data
dTrue = LinRegFuncs( 'dhatFunc', G, mTrue);
data  = LinRegFuncs('NoisyData', G, mTrue, sigma);

%% find the least squares solution and model parameter covariance matrix

mlsq = (G'*G)\(G'*data);
covm = sigma^2*((G'*G)\eye(Nvars));

%% bounds of model and define functions

mNames = {'a','b'};     % model parameter names
mbnds  = [0,1; 0,1];    % model parameter bounds

% function to calculate forward model
% m --> dhat
dhatFunc  = @(model) LinRegFuncs('dhatFunc', G, model);

% function to calculate prior probability given a set of model param values
% m --> prior prob
PriorFunc = @(model) ProbFuncs('PriorFunc', model, mbnds, 'uniform');

% function to sample from the prior probability distributionn
% [] --> set of model parameter values [Niter x Nm]
PrSmpFunc = @(Niter) ProbFuncs('PriorSampFunc', Niter, mbnds, 'uniform');

% function to calculate likelihood of dhat
% dhat --> likelihood 
LikeFunc  = @(dhat) ProbFuncs('LikeFunc',dhat,data,sigma*ones(size(data)));

% function to calculate likelihood from model parameter values
% model --> dhat --> likelihood
LkMdFunc  = @(model) ProbFuncs('LikeFuncModel', dhatFunc, model, data, sigma*ones(size(data)));


%% run one iteration of the model to ensure functions are correct

mtmp = rand(Nvars,1);
[dhat, flag] = dhatFunc(mtmp);
mtmpsmp = PrSmpFunc(10);
pr = PriorFunc(mtmp);
lk = LikeFunc(dhat);
[lm,dhat] = LkMdFunc(mtmp);

%% plot the synthetic data

% sort X for plotting
[Xsort, iX] = sort(x);

figure;
plot(x    , data         , '^'); hold on;
plot(x(iX), G(iX,:)*mTrue, '-' , 'linewidth', 4);
plot(x(iX), G(iX,:)*mlsq , '--', 'linewidth', 4);
plot(x(iX), G(iX,:)*mtmp , '-' );
xlabel('independent variable');
ylabel('dependent variable');
legend('noisy data','true data','Least squares solution','random model','location','best');
legend boxoff;
drawnow;

%% NOW, test different inversion schemes
% establish some common traits so that we can compare the methods

Niter = 1000000;
Nbins = min(500,Niter/20);

%% run MCMC
% adjust step size to get reasonable acceptance ratio ~26%
m0     = rand(Nvars,1);
mstep  = 0.015*diff(mbnds,[],2); % good one
BurnIn = 0.1*Niter;

tic;
[m_mcmc,P_mcmc,count] = mcmc(dhatFunc,PriorFunc,LikeFunc,m0,mstep,mbnds,Niter);
RunTime(1) = toc;

% plot mcmc outputs
xMAP = plotmcmc(m_mcmc, P_mcmc, [], mbnds, count, BurnIn, mNames);
plotcorner(m_mcmc, P_mcmc, m0, mbnds, count, BurnIn, mNames); drawnow;

% retrieve distributions
[ppd_mcmc.m, ppd_mcmc.prob] = CalcPDF(mbnds, m_mcmc(BurnIn:end,:), Nbins);

%% catmip

Nsteps = 20; % number of tempering steps

cmt = tic;
[m_catmip, p_catmip, dhcm, rtcm, m_catmip_all] = catmip(PriorFunc, PrSmpFunc, LkMdFunc, 'Niter', Niter/Nsteps, 'Nsteps', Nsteps);
RunTime(3) = toc(cmt);

% plot outputs
PlotTemperingSteps(m_catmip_all, mbnds, mNames)

% retrieve distributions
[ppd_catm.m, ppd_catm.prob] = CalcPDF(mbnds, m_catmip, Nbins); drawnow;

%% Neighborhood algorithm

NbrOpts       = LoadNbrOpts;
NbrOpts.Ns    = 50;
NbrOpts.Nr    = 20;
NbrOpts.Niter = 20;
NbrOpts.plot  = 0;
NbrOpts.Ngibbs= 1000;
NbrOpts.Nchain= 2;

% search
nbt = tic;
[mReal, mNorm, ~, L, ~, ~, ~] = NeighborhoodSearch('main',...
    dhatFunc, LikeFunc, mbnds, mNames, [], NbrOpts, 1);

% appraise
[~, mOut, mRealOut] = GibbsSampler('main', mNorm, mbnds, L, NbrOpts);

% retrieve distributions
[ppd_nbrh.m, ppd_nbrh.prob] = CalcPDF(mbnds, mRealOut, Nbins);
RunTime(4) = toc(nbt);

% plot
PlotNAIterations(mNorm, mNames, mbnds, L, NbrOpts, 1:2:20)
AddTrueModelToPlot((mTrue-mbnds(:,1))./diff(mbnds,[],2)); drawnow;

%% gwmcmc

if isfolder('../../gwmcmc')
    addpath('../../gwmcmc/');
    
    minit = mbnds(:,1) + diff(mbnds,[],2).*rand(Nvars,100);
    tic
    [m_gw, p_gw] = gwmcmc(minit,{PriorFunc LkMdFunc},Niter,'ThinChain',1,'BurnIn',0.2);
    RunTime(2) = toc;
    
    % retrieve distributions
    m_gw = m_gw(:,:)'; p_gw = p_gw(:,:)'; dhat = dhat(:,:)';
    [ppd_gwmc.m, ppd_gwmc.prob] = CalcPDF(mbnds, m_gw, Nbins);
end

%% compare pdfs of all the schemes

disp(RunTime);

% calculate true probability distribution of model parameters
% from the covariance matrix of the model parameters 
clear ppd_true;
for mi = 1:length(mNames)
    ppd_true.m(:,mi)    = linspace(mbnds(mi,1), mbnds(mi,2), 1000);
    ppd_true.prob(:,mi) = normpdf(ppd_true.m(:,mi), mlsq(mi), sqrt(covm(mi,mi)));
end

figure;
set(gcf,'defaultlinelinewidth', 2, 'Position', [500,600,1000,500]);
colors = lines(4);

for mi = 1:Nvars
    subplot(1,Nvars,mi);
    plot(ppd_true.m(:,mi), ppd_true.prob(:,mi), 'k-', 'LineWidth', 4);
    hold on;
    plot(mTrue(mi)*ones(1,2), ylim, 'k-');
    plot( mlsq(mi)*ones(1,2), ylim, 'k:');
    
    plot(ppd_mcmc.m(:,mi), ppd_mcmc.prob(:,mi), 'Color', colors(1,:));
    plot(ppd_catm.m(:,mi), ppd_catm.prob(:,mi), 'Color', colors(2,:));
    plot(ppd_nbrh.m(:,mi), ppd_nbrh.prob(:,mi), 'Color', colors(3,:));
    
    if isfolder('../../gwmcmc')
        plot(ppd_gwmc.m(:,mi), ppd_gwmc.prob(:,mi), 'Color', colors(4,:)); 
    end
    
    hold off;
    
    leg = legend('True distrib','True value','Least Squares','MCMC','CATMIP','NBR','GWMCMC','location','southoutside');
    legend boxoff
    title(leg, 'Inversion method');
    
    xlim(mlsq(mi) + [-1,1]*10*sqrt(covm(mi,mi)));
    title(mNames{mi});
end


