function varargout = ProbFuncs(varargin)
% functions to calculate prior distribution and likelihood for various
% inversion schemes.
% 
% Ying-Qi Wong, Nov 21, 2019. 

[varargout{1:nargout}] = feval(varargin{:});
end


function [prior, junk] = PriorFunc (model, mu, sigma, distrib)
% prior = ProbFuncs('PriorFunc', model, mu, sigma, distrib)

model = model(:); % just to make sure that model is Nvar x 1
Nvar = length(model);

pvar = zeros(size(model));

switch distrib
    case 'Normal'
        % here mu = mean (Nvar x 1), sigma = standard deviation (Nvar x 1)
        for ivar = 1:Nvar
            pvar(ivar) = normalPrior(model(ivar), mu(ivar), sigma(ivar));
        end
    case 'Uniform'
        % here mu = bounds (Nvar x 1)
        pvar = double(model>=mu(:,1) & model<=mu(:,2));
end

prior = sum(log(pvar));

junk  = nan; % pointless output that is needed for gwmcmc
end

function [models] = PriorSampFunc (distrib, Niter, mdef)
% samples from the prior distribution - either a normal or uniform
% distribution.
% mdef is size (Nvar x 2 - 2 parameters needed to define the distribution)

Nvar = size(mdef,1);

switch distrib
    case 'Normal'
        mu  = repmat(mdef(:,1)', Niter, 1);
        sig = repmat(mdef(:,2)', Niter, 1);
        models = normrnd(mu, sig);
        
    case 'Uniform'
        lower = repmat(mdef(:,1)', Niter, 1);
        upper = repmat(mdef(:,2)', Niter, 1);
        models = unifrnd(lower, upper);
end

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