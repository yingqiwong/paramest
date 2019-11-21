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

% run a least squares
mlin = (G'*G)\(G'*data);

%% bounds of model and define functions

mNames = {'a','b'};
mbnds  = [0,1; 0,1];

dhatFunc  = @(model) LinRegFuncs('dhatFunc', G, model);
PriorFunc = @(model) ProbFuncs('PriorFunc',model,mbnds,[],'Uniform');
LikeFunc  = @(dhat, Linputs) ProbFuncs('LikeFunc',dhat,data,sigma*ones(size(data)),Linputs);
LkMdFunc  = @(model) ProbFuncs('LikeFuncModel', dhatFunc, model, data, sigma*ones(size(data)));


%% run one iteration of the model to ensure functions are correct

xtmp = rand(Nvars,1);
[dhat, ~, Lin] = dhatFunc(xtmp);
pr = PriorFunc(xtmp);
lk = LikeFunc(dhat, Lin);
lm = LkMdFunc(xtmp);

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
Niter = 100000;
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
[mgw, pgw] =gwmcmc(minit,{PriorFunc LkMdFunc},Niter,'ThinChain',1, 'BurnIn',0.2);
RunTime(2) = toc;

mgw = mgw(:,:)'; pgw = pgw(:,:)';
ppd_gw = CalcPPD(mgw, mbnds, 1000);
figure; ecornerplot(mgw(:,:)','ks',true,'color',[.6 .35 .3])


%% Neighborhood algorithm

NbrOpts       = LoadNbrOpts;
NbrOpts.Ns    = 100;
NbrOpts.Nr    = 50;
NbrOpts.Niter = 19;
NbrOpts.plot  = 0;

% search
tic;
[mReal, mNorm, ~, L, ~, ~, ~] = NeighborhoodSearch('main',...
    dhatFunc, LikeFunc, mbnds, mNames, NbrOpts, 1);
RunTime(3) = toc;

% appraise
[ppd_nbr, mOut, mRealOut, LPxi, mChain] = GibbsSampler('main', mNorm, mbnds, L, NbrOpts);
ppd_nbr = CalcPPD(mRealOut, mbnds, 1000);

PlotPosteriorPDF(mRealOut, mbnds, mNames, mTrue);

%% compare pdfs of all the schemes

disp(RunTime);

figure; 
set(gcf,'defaultlinelinewidth', 2, 'Position',[511   667   971   318]);

for i = 1:Nvars
    subplot(1,Nvars,i);
    plot(ppd_mcmc.m(:,i), ppd_mcmc.prob(:,i));
    hold on;
    plot(ppd_gw.m(:,i),   ppd_gw.prob(:,i));
    plot(ppd_nbr.m(:,i),  ppd_nbr.prob(:,i));
    plot(mTrue(i)*ones(1,2), ylim, 'k:');
    hold off;
    leg = legend('MCMC','GWMCMC','NBR1'); legend boxoff;
    title(leg, 'NA search max iter');
end















