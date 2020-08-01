function varargout = LinRegFuncs(varargin)
% Simple functions to predict a linear forward model and generates noisy 
% data, used to test linear regressions
% 
% Ying-Qi Wong, Nov 21, 2019
[varargout{1:nargout}] = feval(varargin{:});
end

function [dhat, flag, Linputs] = dhatFunc (G, model)
% [dhat, flag, Linputs] = LinRegFuncs('dhatFunc', G, model)
% FYI flag and Linputs are just placeholder values needed for some
% inversion schemes.

dhat{1} = G*model(:);
flag = 1;
Linputs = 1;
end

function data = NoisyData (G, model, sigma)
% data = LinRegFuncs('NoisyData', G, model, sigma)

dhat = dhatFunc(G, model);
data = dhat{1} + sigma.*randn(size(dhat{1}));

end

