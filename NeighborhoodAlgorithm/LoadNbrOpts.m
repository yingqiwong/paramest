function [NbrOpts] = LoadNbrOpts ()
% loads a default structure for alteration

% for search
NbrOpts.Ns      = 10;   % number of new samples at each interation (scalar)
NbrOpts.Nr      = 5;    % number of cells to sample at each iteration (scalar)
NbrOpts.Niter   = 2;    % resampling interations
NbrOpts.save    = 0;    % whether to save an output textfile of models
NbrOpts.filename = '';  % if (save), need to have a filename
NbrOpts.Parallel = 0;   % if want to run models in parallel

% for appraisal
NbrOpts.plot     = 1;                   % whether to plot probabilities
NbrOpts.Ngibbs   = 1000;                % number of Gibbs-resampled points
NbrOpts.Nppd     = NbrOpts.Ngibbs/10;   % number of points for ppd
NbrOpts.Nchain   = 2;                  % number of Gibbs chains
end