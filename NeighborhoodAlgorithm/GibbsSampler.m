function varargout = GibbsSampler (varargin)
[varargout{1:nargout}] = feval(varargin{:});
end

function [ppd, mOut, mRealOut, LPxi, mChain] = main (mEn, mBnds, LP, NbrOpts)
% Appraises the model ensemble using the Neighborhood Appraisal Algorithm
% Sambridge II, 1999.
% 
% Inputs
% mEn       ensemble of models (Niter x Nvar, should be normalized [0,1])
% mBnds     bounds on model params (Nvar x 2)
% LP        log likelihood of models (Niter x 1)
% NbrOpts   options (Nchain, Nrs, Nppd)

% Outputs
% ppd       posterior probability density function 
% mOut      matrix of resampled points in normalized units (Nout x Nvar)
% mRealOut  matrix of resampled points in real units (Nout x Nvar)
% LPxi      log likelihood of resampled points
% mChain    individual Gibbs sampler chains with log probabilities

addpath([fileparts(which(mfilename)) '/EvalFuncs/']);

Nchain = NbrOpts.Nchain;
Nrs    = NbrOpts.Ngibbs;
Nppd   = NbrOpts.Nppd;

Nvar        = size(mEn,2);

mChain      = cell(Nchain,2);
% parpool(Nchain);
for ich = 1:Nchain
    fprintf('Evaluating chain %d of %d.\n', ich, Nchain);
    
    Chain   = zeros(Nrs, Nvar);
    LPChain = zeros(Nrs, 1);
    
    tic;
    [Chain(1,:), LPChain(1)] = RandomWalkAllDim(mEn, LP, rand(1,Nvar));
    for in = 2:Nrs
        [Chain(in,:), LPChain(in)]  = RandomWalkAllDim(mEn, LP, Chain(in-1,:));
    end
    toc;
    
    mChain{ich,1} = Chain;
    mChain{ich,2} = LPChain;
end

mOut = cell2mat(mChain(:,1));
LPxi = cell2mat(mChain(:,2));

mRealOut = ConvertToRealUnits(mOut, mBnds);

fprintf('Running ks density...\n');
ppd.m    = zeros(Nppd, Nvar);
ppd.prob = zeros(Nppd, Nvar);
for ivar = 1:Nvar
    ppd.m(:,ivar)    = linspace(mBnds(ivar,1), mBnds(ivar,2),Nppd)';
    ppd.prob(:,ivar) = ksdensity(mRealOut(:,ivar),ppd.m(:,ivar));
end
fprintf('Finished running ks density.\n');

end

function [xANew, LPxi, iter] = RandomWalkAllDim (mEn, LP, xA)
% performs random walk along all variable directions, where the stairstep
% Assumes that range of variables xA, xANew is [0,1]
% 
% Inputs
% mEn       ensemble of models (Niter x Nvar)
% LP        log probability of models (Niter x 1)
% xA        starting point (Nvar x 1)
%
% Outputs
% xANew     next point (all dimensions looped through)
% LPxi      log probability of resampled points
% iter      number of iterations needed in single dimension random walk

Nvar = length(xA);

xANew = xA;
iter  = zeros(Nvar,1);

for ivar = 1:Nvar
    [xji, xcell] = CalcIntersectionsAlongAxis(mEn, xANew, ivar);
    [xANew(ivar), LPxi, iter(ivar)] = RandomWalkOneDim(xji, LP(xcell), xANew(ivar));
end

end


function [xANew, LPxi, iter] = RandomWalkOneDim (xji, LP, xA)
% performs random walk along one variable direction, where the stairstep
% conditional PDF is defined by the intervals xji and probabilities P. 
% Assumes that range of variables xA, xANew is [0,1]
% 
% Inputs
% xji       Intervals of the conditional PDF (Nint x 2)
% LP        log probability within each interval (Nint x 1)
% xA        Initial value of the variable (scalar)
% 
% Outputs
% xANew     Random walk step 
% LPxi      log probability at xANew
% 
% possibly add a MaxIter function so that we don't get stuck in the while
% loop

xANew = xA;

Pmax = max(LP);
LPxi = GetConditionalProbability(xji, LP, xA);

% set initial r to be greater than 1 so the loop takes at least one step.
r = 1.2;
iter = 0;

while r > (LPxi-Pmax)
    iter  = iter + 1;
    xANew = rand(1);    % random walk
    LPxi  = GetConditionalProbability(xji, LP, xANew);
    r     = log10(rand(1));
end


end

function [LPxi] = GetConditionalProbability (xji, LP, xA)
% Returns conditional probability from stairstep plot
% If not using Neighborhood Algorithm for direct search, this is the only
% function that needs to be changed
% 
% Inputs
% xji       Intervals of the conditional PDF (Nint x 2)
% P         log probability within each interval (Nint x 1)
% xA        Initial value of the variable (scalar)
% 
% Outputs
% Pxi       log probability at xA

LPxi = LP(find(xA<xji(:,2),1));

end