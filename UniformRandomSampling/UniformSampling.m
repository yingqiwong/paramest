function [m, dhat, flag, RunTime] = UniformSampling (...
    dhatFunc, mbnds, mNames, usOpts, Ndata)
% runs a model with uniform random sampling just to get the predicted data
%
%
% INPUTS
% dhatFunc  function handle that predicts data
% mbnds     lower, upper bounds of variables (Nvar x 2)
%           if you want to fix a variable, just set lower = upper
% mNames    parameter names
% usOpts    Options for uniform sampling (includes Ntot)
% Ndata     number of datasets to initialize dhat matrix
%
% OUTPUTS
% mReal     models in real units (Ntot x Nvar)
% dhat      predicted data (Ntot x Ndata)
% flag      whether model ran successfully (Ntot x 1)
% RunTime   runtime for models (Ntot x 1)
%
% YQW, 13 Dec 2020
%

% initialize output matrices
Nvar    = size(mbnds,1);
dhat    = cell(usOpts.Ntot, Ndata);
flag    = nan(usOpts.Ntot,1);
RunTime = flag;

% uniform random sampling for model params
m  = mbnds(:,1)' + diff(mbnds,[],2)'.*rand(usOpts.Ntot, Nvar);

% counter for evaluation blocks before saving
Nloop = 0;

% initialize file
if usOpts.save
    save(usOpts.filename, ...
        'Nloop', 'mNames', 'mbnds', 'usOpts', 'm', 'dhat', 'flag', 'RunTime');
end

% start running models. 
% Note nested loops so we can save data after Nit iterations in outer loop. 
while Nloop < usOpts.Ntot
    
    
    for ins = Nloop+(1:usOpts.Nit)
        
        tic;
        [dhatIter, flag(ins)] = dhatFunc(m(ins,:));
        RunTime(ins) = toc;
        
        if flag(ins)==1, dhat(ins,:) = dhatIter; end
        
    end
    
    % save outputs
    save(usOpts.filename, ...
        'Nloop', 'mNames', 'mbnds', 'usOpts', 'm', 'dhat', 'flag', 'RunTime');
    

    % move to next block of runs
    Nloop = Nloop + usOpts.Nit;
    
end


end