function varargout = ProbFuncs(varargin)
% functions to calculate prior distribution and likelihood for various
% inversion schemes.
% 
% Ying-Qi Wong, Nov 21, 2019. 

[varargout{1:nargout}] = feval(varargin{:});
end


function [prior, junk] = PriorFunc (model, mdef, distrib)
% 
% prior = ProbFuncs('PriorFunc', model, mdef, distrib)
% 
% returns the log - prior probability of a model
% mdef is the matrix of parameters needed to define the prior distribution
% definition below, depends on distrib

switch distrib
    case 'normal'
        % mdef = [mu, sigma]; size Nvar x 2
        mu    = mdef(:,1);
        sigma = mdef(:,2);
        pvar  = normpdf(model(:), mu, sigma);
        
    case 'uniform'
        % mdef = [lower, upper]; size Nvar x 2
        lower = mdef(:,1);
        upper = mdef(:,2);
        pvar  = double(model(:)>=lower & model(:)<=upper);
        
end

prior = sum(log(pvar));

junk  = nan; % pointless output that is needed for gwmcmc
end

function [models] = PriorSampFunc (Niter, mdef, distrib)
% 
% [models] = PriorSampFunc (Niter, mdef, distrib)
% 
% samples from the prior distribution 
% mdef is the matrix of parameters needed to define the prior distribution
% definition below, depends on distrib

switch distrib
    case 'normal'
        % mdef = [mu, sigma]; size Nvar x 2
        mu     = mdef(:,1);
        sigma  = mdef(:,2);
        models = mu +  sigma.*randn(length(sigma),Niter);
        
    case 'uniform'
        % mdef = [lower, upper]; size Nvar x 2
        lower  = mdef(:,1);
        upper  = mdef(:,2);
        models = lower + (upper-lower).*rand(length(lower),Niter);
end

models = models'; % the output size has to be Niter x Nvar

end


function L = LikeFunc (dhat, data, sigma)
% L = ProbFuncs('LikeFunc', dhat, data, sigma)
% assumes normally-distributed errors. 
% 
L = sum(- 0.5*((data(:) - dhat(:))./sigma(:)).^2 );
end


function [L, dhat] = LikeFuncModel (dhatFunc, model, data, sigma)
% calculates likelihood given input model parameters, basically combining
% dhatfunc and likefunc

[dhat, flag] = dhatFunc(model);

if flag == 1
    L = LikeFunc(dhat, data, sigma);
else
    L = nan;
end

end


function [logpdf, gradlogpdf] = logPosterior (model,r,nu,data,sigma,mu_prior,sig_prior)
% for Hamiltonian MC. From the Bayesian Linear Regression Using
% Hamitonian MC tutorial on Mathworks.


% Compute the log likelihood and its gradient (i.e. d(logL)/dmodeli)
dhat     = dhatFunc_exp(model, r, nu);
loglik   = LikeFunc(dhat, data, sigma);

D  = exp(model(2));
durdV = dhat;
durdD = -3*dhat*D.^2./(r.^2+D.^2);
gradlik  = sum((data-dhat)/sigma^2.*[durdV, durdD])' ;

% Compute log priors and gradients
logprior = zeros(size(model));
gradprior = zeros(size(model));

for mi = 1:length(model)
    [logprior(mi), gradprior(mi)] = normalPrior(model(mi), mu_prior(mi), sig_prior(mi));
end

% Return the log posterior and its gradient
logpdf     = loglik + sum(logprior);
gradlogpdf = gradlik + gradprior;
end


function [logpdf,gradlogpdf] = normalPrior(P,Mu,Sigma)
Z          = (P - Mu)./Sigma;
logpdf     = sum(-log(Sigma) - .5*log(2*pi) - .5*(Z.^2));
gradlogpdf = -Z./Sigma;
end