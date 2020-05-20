function varargout = NeighborhoodSearch (varargin)
[varargout{1:nargout}] = feval(varargin{:});
end

function [mReal, mNorm, dhat, L, Lmax, flag, RunTime] = main (...
    dhatFunc, LikeFunc, mbnds, mNames, mIn, NbrOpts, Ndata, varargin)
% evaluates the neighborhood direct search algorithm from Sambridge 1999 I
% assumes uniform prior - should this part allow different priors?
%
% INPUTS
% dhatFunc  function handle that predicts data
% LikeFunc  function handle that calculates likelihood (= - misfit)
% mbnds     lower, upper bounds of variables (Nvar x 2)
%           if you want to fix a variable, just set lower = upper
% mNames    parameter names
% mIn       structure of inputs from previous runs, or EMPTY
%           (includes mNorm, mReal, dhat, L, Lmax, flag, RunTime)
% NbrOpts   Options structure:
%           Ns        number of new samples at each interation (scalar)
%           Nr        number of cells to sample at each iteration (scalar)
%           Niter     resampling interations
%           save      whether to save an output textfile of models
%           filename  if (save), need to have a filename
%           Parallel  if want to run models in parallel
% Ndata     number of datasets (for joint inversions)
% varargin  any other input values for dhatFunc
%
% OUTPUTS
% vReal     models in real units (ns*(nRS+1) x Nvar)
% vNorm     models in normalized units (0-1) (ns*(nRS+1) x Nvar)
% L         log likelihood (ns*(nRS+1) x 1)
% Lmax      max log likelihood(nRS+1 x 1)
% flag      whether model ran successfully (ns*(nRS+1) x 1)
%
% YQW June 24, 2019

% addpath([fileparts(which(mfilename)) '/EvalFuncs/']);
rng('shuffle'); % just to make sure you don't repeat the same seed

Ns    = NbrOpts.Ns;
Nr    = NbrOpts.Nr;
Niter = NbrOpts.Niter;

if (NbrOpts.Parallel)
    NumWorkers = min([NbrOpts.Ncores,NbrOpts.Ns]);
else
    NumWorkers = 0;
end

% some helpful initial analysis
Nvar        = size(mbnds,1);
VarVary     = find(diff(mbnds,[],2)>0);
ns_adjusted = floor(Ns/Nr)*Nr;  % enforce that this is an integer
rst_opt     = ~isempty(mIn);    % whether we are trying to restart a run

% initialize matrices
mNorm  = ones(Ns + ns_adjusted*Niter, Nvar);
mReal  = mNorm;
dhat   = cell(Ns + ns_adjusted*Niter, Ndata);
L      = nan*ones(Ns + ns_adjusted*Niter, 1);
Lmax   = nan*ones(1+Niter,1);
flag   = L;
RunTime= L;

if (rst_opt) 
    % want to restart a run
    Ncurr    = length(mIn.L);
    inr_curr = (Ncurr - Ns)/ns_adjusted;
    
    mNorm(1:Ncurr,:) = mIn.mNorm;
    mReal(1:Ncurr,:) = mIn.mReal;
    dhat(1:Ncurr,:)  = mIn.dhat;
    L(1:Ncurr)       = mIn.L;
    Lmax(1:inr_curr) = mIn.Lmax;
    flag(1:Ncurr)    = mIn.flag;
    RunTime(1:Ncurr) = mIn.RunTime;
    
    [inds, Lmax(inr_curr+1)] = Lsort(L(1:Ncurr));
    fprintf('Max log-likelihood = %.2e.\n', Lmax(inr_curr+1));
else
    % NEW RUN
    % randomly sample model space, but fix variables where upper = lower bounds
    mNorm(1:Ns,VarVary) = rand(Ns,length(VarVary));
    inr_curr = 0;
    
    fprintf('Generating initial %d random samples...\n', Ns);
    [mReal(1:Ns,:), dhat(1:Ns,:), flag(1:Ns), RunTime(1:Ns), L(1:Ns)] = RunForwardModel(...
        dhatFunc, LikeFunc, mNorm(1:Ns,:), mbnds, Ns, Ndata, NumWorkers, varargin{:});
    [inds, Lmax(1)] = Lsort(L(1:Ns));
    fprintf('Max log-likelihood = %.2e.\n', Lmax(1));
    fprintf('Finished generating initial random samples\n');
    
end

if NbrOpts.save
    save(NbrOpts.filename, 'mNames', 'mbnds', 'NbrOpts', ...
        'mNorm', 'mReal', 'dhat', 'L', 'Lmax', 'flag', 'RunTime');
end

% neighborhood sampling
fprintf('Generating %d total samples for %d low misfit cells...\n', Ns, Nr);
for inr = (inr_curr+1):Niter
    fprintf('Neighborhood sampling iteration %d of %d.\n', inr, Niter);
    vInd = Ns + (inr-1)*ns_adjusted;
    
    % obtain new samples
    vNew  = ones(ns_adjusted,Nvar);
    vNew(:,VarVary) = NeighborhoodSampling(mNorm(1:vInd,VarVary), inds(1:Nr), Ns);
    
    [vRealnew, dhatnew, flagnew, RunTimenew, Lnew] = RunForwardModel(...
        dhatFunc, LikeFunc, vNew, mbnds, ns_adjusted, Ndata, NumWorkers, varargin{:});
    
    % fill in matrices of models and misfits
    mNorm(vInd+(1:ns_adjusted),:) = vNew;
    mReal(vInd+(1:ns_adjusted),:) = vRealnew;
    dhat(vInd+(1:ns_adjusted),:)  = dhatnew;
    flag(vInd+(1:ns_adjusted))    = flagnew;
    RunTime(vInd+(1:ns_adjusted)) = RunTimenew;
    L(vInd+(1:ns_adjusted))       = Lnew;
    
    % find maximum likelihood
    [inds, Lmax(1+inr)] = Lsort(L(1:(vInd+ns_adjusted)));
    fprintf('Max log-likelihood = %.2e.\n', Lmax(inr+1));
    
    if NbrOpts.save
        save(NbrOpts.filename, 'mNames', 'mbnds', 'NbrOpts', ...
            'mNorm', 'mReal', 'dhat', 'L', 'Lmax', 'flag', 'RunTime');
    end
    
end
fprintf('Finished neighborhood sampling.\n');

end

function [mReal, dhat, flag, RunTime, L] = RunForwardModel (...
    dhatFunc, LikeFunc, vNorm, mbnds, Ns, Ndata, NumWorkers, varargin)
% use this function after neighborhood sampling
% transforms new normalized samples to real units and runs the forward
% model and calculates likelihoods

dhat    = cell(Ns,Ndata);
flag    = nan*ones(Ns,1);
RunTime = nan*ones(Ns,1);
L       = nan*ones(Ns,1);

mReal = TransformToRealUnits(vNorm, mbnds);

if NumWorkers>0, parpool(NumWorkers); end

parfor (ins = 1:Ns, NumWorkers)
    
    df = dhatFunc;
    Lf = LikeFunc;
    inputs = varargin;
    
    tic;
    [dhatIter, flag(ins), Linputs] = df(mReal(ins,:), inputs{:});
    RunTime(ins) = toc;
    
    if flag(ins)==1
        dhat(ins,:) = dhatIter;
        L(ins)      = Lf(dhatIter, Linputs);
    end
end

end

function [inds, Lmax] = Lsort (L)
% sort the likelihoods (handy when changing between L and misfit)
% sometimes L may return an imaginary number - faulty. Sort according to
% real values only.
[~,inds] = sort(L,'descend','MissingPlacement','last','ComparisonMethod','real');
Lmax     = L(inds(1));
end

function [mReal] = TransformToRealUnits (mNorm, bnds)
% bnds: lower and upper bounds of variables (Nvars x 2)

bnds  = bnds';
mReal = bnds(1,:) + (bnds(2,:) - bnds(1,:)).*mNorm;

end

function [NewSamps] = NeighborhoodSampling (v, k_ind, ns)
% samples Voronoi cells to generate new samples for the next misfit
% evaluation. 
% All samples are of range (0,1). Transformation to real units is done
% outside this function to calculate dhat and likelihood.
% 
% INPUTS
% x     coordinates of points (Npts x Nvars)
% k_ind index of cells that you want to generate new samples for (nr x 1)
% ns    total number of new samples (scalar)
% 
% OUTPUTS
% NewSamps  Matrix of new samples (N*nr x Nvars)
% 

nr   = length(k_ind);
nsnr = floor(ns/nr); % number of new samples per cell

Nvar     = size(v,2);
NewSamps = zeros(nr*nsnr, Nvar);


for k = 1:nr         % loop through voronoi cells
    xA = v(k_ind(k),:);
    
    for in = 1:nsnr  % loop through new samples for each voronoi cell
        order = randperm(Nvar);
        
        for ivar = order % loop through variables using different orders
            xA = NewSample(v, xA, ivar);
        end
        
        NewSamps((k-1)*nsnr+in, :) = xA;
    end
end


end

function [xAnew] = NewSample (v, xA, ivar)
% generates new samples in the ivar-th variable (analogous to axis)

[Npts, ~] = size(v);

% find nearest cell node by minimizing distance between v and xA
ds     = sum((v - xA).^2, 2); 
[~, k] = min(ds);
vk     = v(k,:);

dk2 = norm((vk - xA)).^2 - (vk(ivar)-xA(ivar)).^2;

xji = nan(Npts, 1);

for ix = 1:Npts
    
    if ix == k, continue; end

    dj2 = norm((v(ix,:) - xA)).^2 - (v(ix,ivar) - xA(ivar)).^2;
    xji(ix) = 0.5*(vk(ivar)+v(ix,ivar) + (dk2 - dj2)/(vk(ivar)-v(ix,ivar)));
    
end

lower = max([xji(xji<=xA(ivar)); 0]);
upper = min([xji(xji>=xA(ivar)); 1]);

xAnew = xA;
xAnew(ivar) = lower + (upper - lower)*rand(1);

end
