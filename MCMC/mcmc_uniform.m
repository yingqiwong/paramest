function [x_keep,L_keep,count] = mcmc(func,data,x0,xstep,xbnds,sigma,Niter,varargin)
%this function computes Markov Chain Monte Carlo sampling using random walk
%and Metropolis Hastings with uniform prior and normally-distributed data errors.
% Ying-Qi Wong, 2 Dec 2014
%
%Inputs:
%     func  = name of function that predicts data from model params
%     data  = vector of observations
%     x0    = initial estimate of parameter vector (Nx1)
%     xstep = step size in all parameter directions (Nx1)
%     xbnds = Nx2 matrix of lower and upper bounds
%     sigma = sigma of normal distribution of data
%     Niter = number of iterations
%     varargin = stores the parameters required for the function func
%  
%Outputs:
%     x_keep = array of samples
%     L_keep = likelihood of samples
%     count  = number of accepted.  Acceptance ratio is count/Niter


%Analyze inputs
model_size = length(x0);        %find number of model parameters
x1 = x0;                        %set initial guess as first candidate
fun = fcnchk(func);             %check input function  

%evaluate input function to get predicted data and calculate the
%probability of this model given the data using a Bayesian approach.
%Assumptions: normally-distributed, uncorrelated data errors
dprop = fun(x1, varargin{:});       
% Pd_x1 = exp(-1/2/(sigma^2)*(data-dprop)'*(data-dprop)); %P(d|x1)
Pd_x1 = exp(-0.5*sum(((data - dprop)./sigma).^2));
%Pd_x1 = normpdf(sum(data-dprop),0,sigma);
P_x = 1;                                                %P(x) assume uniform
Px1_d = Pd_x1 * P_x;                                    %P(x1|d)

%Initialize the vectors x_keep and L_keep to store the accepted models
count=0;
x_keep = zeros(model_size,Niter);
L_keep = zeros(Niter,1);

%Begin loop to perform MCMC
for i=1:Niter
    %Random walk chain to find the proposed model x2
    x2 = x1-xstep + 2*xstep.*rand(model_size,1);
    
    %Check that the proposed model falls within the bounds.  If it falls
    %outside the bounds, then the loop is broken and we go back to the
    %beginning to find a new proposed model.
    for k=1:model_size
        if x2(k)<xbnds(k,1) || x2(k)>xbnds(k,2)
            out_bnds = 1;
            break;
        else
            out_bnds = 0; 
        end
    end
    
    if out_bnds==1
        continue;   %go to the next iteration of the MCMC for loop
    end

    %Evaluate forward model for the proposed model x2 that is within bounds
    dprop = fun(x2, varargin{:});

    %Evaluate probability of the model given the data using Bayesian
    %approach: P(x2|d)~ P(d|x2) P(x2)
    Pd_x2 = exp(-0.5*sum(((data - dprop)./sigma).^2));
    %Pd_x2 = normpdf(sum(data-dprop),0,sigma);
    Px2_d = Pd_x2 * P_x;

    %Analyze the acceptance criterion by taking the ratio of the
    %probability of the proposed model to the previous model and comparing
    %this probability to a random number between 0 and 1.
    P_accept = min(log(Px2_d/Px1_d),0);
    u = log(rand(1));   %some random number on the interval [0,1]
    
    if u<=P_accept          %i.e. the model is accepted
        x_keep(:,i) = x2;   %store the proposed model and its likelihood
        L_keep(i) = Px2_d;
        x1 = x2;            %assign the accepted model for the next comparison
        Px1_d = Px2_d;
        count = count+1;
    else
        x_keep(:,i) = x1;
        L_keep(i) = Px1_d;
    end
end
   
x_keep = x_keep';
end
