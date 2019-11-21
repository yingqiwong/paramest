function [x_keep,P_keep,count] = mcmc(dhatFunc,PriorFunc,LikeFunc,x0,xstep,xbnds,Niter,varargin)
% this function computes Markov Chain Monte Carlo sampling using random walk
% and Metropolis Hastings. Allows any sort of distribution for prior and
% likelihood
% Ying-Qi Wong, 3 Sep 2019
%
% Inputs:
% dhatFunc  = function that predicts data from model params
% priorFunc = function that evaluates log prior probabilities 
% likeFunc  = function that calculates log data likelihood
% x0        = initial estimate of parameter vector (Nvar x 1)
% xstep     = step size in all parameter directions (Nvar x 1)
% xbnds     = Nx2 matrix of lower and upper bounds (can be empty)
% Niter     = number of iterations
% varargin  = any other input values for dhatFunc
% 
% NB: priorFunc and likeFunc should only take one input each: the model
% parameter vector and the data vector respectively.
%  
%Outputs:
% x_keep = array of samples (Nvar x Niter)
% P_keep = posterior distribution
% count  = number of accepted.  Acceptance ratio is count/Niter


%Analyze inputs
Nvar  = length(x0);        %find number of model parameters
x1    = x0;                  %set initial guess as first candidate
xstep = xstep(:);
if isempty(xbnds), xbnds = [-Inf, Inf]; end

dhatFunc  = fcnchk(dhatFunc);         
PriorFunc = fcnchk(PriorFunc);
LikeFunc  = fcnchk(LikeFunc);

% evaluate first model
[dhat, dflag, Linputs] = dhatFunc(x1, varargin{:});    
Px1   = PriorFunc(x1);
Ld_x1 = LikeFunc(dhat, Linputs);
Px1_d = Px1 + Ld_x1;

%Initialize the vectors x_keep and L_keep to store the accepted models
count  = 0;
x_keep = zeros(Nvar,Niter);
P_keep = zeros(Niter,1);
Nprint = floor(Niter/20);

%Begin loop to perform MCMC
for i=1:Niter
    
    % print the progress
    if mod(i,Nprint)==0, fprintf('Iteration %d of %d.\n', i, Niter); end
    
    flag = 0;
    
    while flag == 0
        %Random walk chain to find the proposed model x2
        x2 = x1-xstep + 2*xstep.*rand(Nvar,1);
        
        %Check that the proposed model falls within the bounds.  If it falls
        %outside the bounds, go back to the beginning of the while loop to
        %search for a new model
        if all(x2>=xbnds(:,1)) && all(x2<=xbnds(:,2)), flag = 1; end
    end
    
    %Evaluate forward model for the proposed model x2 that is within bounds
    [dhat, dflag, Linputs] = dhatFunc(x2, varargin{:});
    if dflag~=1, continue; end

    %Evaluate probability of the model given the data using Bayesian
    %approach: P(x2|d)~ L(d|x2) P(x2)
    Px2   = PriorFunc(x2);
    Ld_x2 = LikeFunc(dhat, Linputs);
    Px2_d = Px2 + Ld_x2;

    %Analyze the acceptance criterion by taking the ratio of the
    %probability of the proposed model to the previous model and comparing
    %this probability to a random number between 0 and 1.
    P_accept = min((Px2_d-Px1_d),0);
    u = log(rand(1));   %some random number on the interval [0,1]
    
    if u<=P_accept          %i.e. the model is accepted
        x_keep(:,i) = x2;   %store the proposed model and its likelihood
        P_keep(i) = Px2_d;
        x1 = x2;            %assign the accepted model for the next comparison
        Px1_d = Px2_d;
        count = count+1;
    else
        x_keep(:,i) = x1;
        P_keep(i) = Px1_d;
    end
end
   
x_keep = x_keep';
end
