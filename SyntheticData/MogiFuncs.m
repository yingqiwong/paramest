function varargout = MogiFuncs(varargin)
% Simple functions to predict surface radial displacements from a Mogi
% source given volume change dV and depth D. Also creates a model with
% noise for synthetic data.
% 
% Depending on the nature of the inversion scheme, you may want dV, D in
% log space. In that case, use dhatFunc_exp.
% 
% Ying-Qi Wong, Nov 21, 2019

[varargout{1:nargout}] = feval(varargin{:});
end

function [dhat, flag, Linputs] = dhatFunc (model, r, nu)
% [dhat, flag] = MogiFuncs('dhatFunc', model, r, nu)

dV = (model(1));
D  = (model(2));
dhat = 1/pi*(1-nu)*dV*(r./(r.^2 + D^2).^(1.5));

flag = 1;
Linputs = 0;

end

function [dhat, flag, Linputs] = dhatFunc_exp (model, r, nu)
% [dhat, flag] = MogiFuncs('dhatFunc_exp', model, r, nu)

dV = exp(model(1));
D  = exp(model(2));
dhat = 1/pi*(1-nu)*dV*(r./(r.^2 + D^2).^(1.5));

flag = 1;
Linputs = 0;
end

function data = NoisyData (model, r, nu, sigma)
% data = MogiFuncs('NoisyData', model, r, nu, sigma)

% toggle choice of dhatFunc to match form of dV, D
dhat = dhatFunc(model, r, nu);
% dhat = dhatFunc_exp(model, r, nu)

data = dhat + sigma*randn(length(r),1);
end
