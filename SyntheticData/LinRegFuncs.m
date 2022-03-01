function varargout = LinRegFuncs(varargin)
% Simple functions to predict a linear forward model and generates noisy 
% data, used to test linear regressions
% 
% Ying-Qi Wong, Nov 21, 2019
[varargout{1:nargout}] = feval(varargin{:});
end

function [dhat, flag] = dhatFunc (G, model)
% [dhat, flag] = LinRegFuncs('dhatFunc', G, model)
% 
% FYI flag is a placeholder values needed for some inversion schemes to
% show that the model had run successfully

dhat = G*model(:);
flag = 1;

end

function data = NoisyData (G, model, sigma)
% data = LinRegFuncs('NoisyData', G, model, sigma)
% put some noise in the data


dhat = dhatFunc(G, model);
data = dhat + sigma.*randn(size(dhat));

end

