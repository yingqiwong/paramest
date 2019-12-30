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
mTrue = [5;6];
sigma = 2e-3;

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
%% MCMC
% adjust step size to get reasonable acceptance ratio ~26%

x0    = rand(Nvars,1);
xstep = 0.01*diff(mbnds,[],2); 
Niter = 10000;
BurnIn = 0.1*Niter;

tic;
[x_keep,P_keep,count] = mcmc(dhatFunc,PriorFunc,LikeFunc,x0,xstep,mbnds,Niter);
RunTime(1) = toc;
fprintf('Acceptance ratio = %.2f.\n', count/Niter*100);

xMAP = PlotMCMCAnalytics(x_keep, P_keep, mbnds, count, BurnIn, mNames);
figure; ecornerplot(x_keep,'ks',true,'color',[.6 .35 .3])
ppd_mcmc = CalcPPD(x_keep(BurnIn:end,:), mbnds, 1000);

%% gwmcmc

minit = mbnds(:,1) + diff(mbnds,[],2).*rand(Nvars,100);

tic
[mgw, pgw, dhat] =gwmcmc(minit,{PriorFunc LkMdFunc},Niter,'ThinChain',1, ...
    'BurnIn',0.2,'OutputData',1,'FileName','','ProgressBar',false);
RunTime(2) = toc;

mgw = mgw(:,:)'; pgw = pgw(:,:)'; dhat = dhat(:,:)';
ppd_gw = CalcPPD(mgw, mbnds, 1000);
figure; ecornerplot(mgw(:,:)','ks',true,'color',[.6 .35 .3])

figure;
iplt=randi(length(dhat),1,100);
for kk=iplt
    h=plot(r(ir),dhat{kk,2}(ir),'color',[.6 .35 .3].^.3);
    hold on
end
hdat  = errorbar(r, data, sigma*ones(size(data)), '^','color',lines(1)); hold on;
htrue = plot(r(ir), dTrue{1}(ir), 'r-', 'linewidth', 4);
xlabel('x_1'); ylabel('y');
legend([hdat;htrue],{'Noisy';'True'},'location','best'); 
legend boxoff;
drawnow;

%% catmip

Ncm = 500;
mtmp = PrSmpFunc(10);
tic
[xcm, LLK, dhcm, allx] = catmip(PrSmpFunc, LkMdFunc, mbnds...
    'Niter', Ncm, 'Nsteps', 5);
RunTime(3) = toc;
ppd_catmip = CalcPPD(xcm, mbnds, 1000);
figure; ecornerplot(xcm,'ks',true,'color',[.6 .35 .3])
PlotTemperingSteps(allx, mbnds, mNames)

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

%% Neighborhood algorithm

NbrOpts       = LoadNbrOpts;
NbrOpts.Ns    = 100;
NbrOpts.Nr    = 50;
NbrOpts.Niter = 19;
NbrOpts.plot  = 0;
NbrOpts.Ngibbs= 200;

% search
tic;
[mReal, mNorm, ~, L, ~, ~, ~] = NeighborhoodSearch('main',...
    dhatFunc, LikeFunc, mbnds, mNames, NbrOpts, 1);
RunTime(4) = toc;

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
    plot(ppd_catmip.m(:,i), ppd_catmip.prob(:,i));
%     plot(ppd_nbr.m(:,i),  ppd_nbr.prob(:,i));
    plot(mTrue(i)*ones(1,2), ylim, 'k:');
    hold off;
    leg = legend('MCMC','GWMCMC','CATMIP','NBR1'); legend boxoff;
    title(leg, 'NA search max iter');
end















