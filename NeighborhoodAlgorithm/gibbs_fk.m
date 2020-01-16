function [ppd, mOut] = gibbs_fk (mEn, misfit, mbnds, NbrOpts)
% using Yo Fukushima's gibbs sampler

PTS = [mEn, misfit];
Nvar = size(mEn, 2);

mChain   = cell(NbrOpts.Nchain,2);

for loopvar = 1:NbrOpts.Nchain
    out = nabparforloop(loopvar,PTS,NbrOpts.Ngibbs,Nvar,mbnds(:,2),mbnds(:,1));
    mChain(loopvar,:) = {out(:,1:Nvar), out(:,end)};
end

mOut = cell2mat(mChain(:,1));


fprintf('Running ks density...\n');
ppd.m    = zeros(NbrOpts.Nppd, Nvar);
ppd.prob = zeros(NbrOpts.Nppd, Nvar);
for ivar = 1:Nvar
    ppd.m(:,ivar)    = linspace(mbnds(ivar,1), mbnds(ivar,2),NbrOpts.Nppd)';
    ppd.prob(:,ivar) = ksdensity(mOut(:,ivar), ppd.m(:,ivar));
end
fprintf('Finished running ks density.\n');
end

function [mEnOut, LPOut] = CleanEnsemblePts (mEnIn, LPIn)
% clean up the input ensemble of models

mEnOut = mEnIn;

% remove unevaluated points
mEnOut(isnan(LPIn),:) = [];

% remove models that are out of bounds
RmFlag = zeros(size(mEnOut,1),1);
for mi = 1:size(mEnOut,1)
    RmFlag(mi) = (any(mEnOut(mi,:) < 0)) || (any(mEnOut(mi,:)) > 1);
end
mEnOut(RmFlag==1,:) = [];

% sort by decreasing log-probability
tmp = sortrows([LPIn, mEnOut], 'descend');
LPOut  = tmp(:,1);
mEnOut = tmp(:,2:end);
end

function outpts=nabparforloop(loopvar,PTS,npts,NDIM,maxlim,minlim)

index=loopvar;
xb = PTS(index,:); % current point
outpts=[];
for np = 1:npts
    
    % Select the component randomly, i.e., for every deviate, the
    % order of the components that the random walk is executed is
    % different.
    rand('state',sum(100*clock));
    odr = randperm(NDIM);
    
    for n = 1:NDIM
        i = odr(n);
        
        cell_minval = [];
        cell_maxval = [];
        Intersect_cell_pt1 = [];
        Intersect_cell_pt2 = [];
        
        %% climb up toward max bound to collect the Voronoi cells %%
        %% which intersect with the current axis. %%
        
        % climb up one by one, until gets to the maximum bound
        % (when ind becomes empty)
        ind = 0; minval = xb(i);
        vk = PTS(index,:); % the node of the voronoi cell in which xb resides
        dj = []; % perpendicular distances to the current axis
        
        while ~isempty(ind)
            x = []; % the coordinates of the intersection
            
            %% calculate the intersection points
            %% using Sambridge (1999) eq.19,24
            % Squared distance between index point and lowest misfit point
            % along all axis in the parameter space except the ith
            dk = sum((vk(1:NDIM)-xb(1:NDIM)).^2) - (vk(i)-xb(i)).^2;
            for j = 1:size(PTS,1)
                if ind==0 % calc. dj only for the first loop
                    dj(j) = sum((PTS(j,1:NDIM)-xb(1:NDIM)).^2) - (PTS(j,i)-xb(i)).^2;
                end
                if vk(i) == PTS(j,i)
                    x(j) = NaN;
                else
                    % Sambridge (1999a) eq. 19, all the intersection points
                    x(j) = 1/2 .* (vk(i)+PTS(j,i)+((dk-dj(j))./(vk(i)-PTS(j,i))));
                end
            end
            
            % retrieve the cell number which is next to the
            % previous one and the intersection point
            ind = find(x > minval & x < 1);
            if ~isempty(ind)
                % boundary of the current Voronoi cell, taille et
                % localisation dans la liste des pts
                [minval, Ind] = min(x(ind));
                cell_minval = [cell_minval; ind(Ind), minval];
                vk = PTS(ind(Ind),:);
            end
        end
        
        % change Intersect_cell_pt1 in such a way that it
        % contains the intervals
        if ~isempty(cell_minval)
            Intersect_cell_pt1(:,1) = cell_minval(:,1);
            Intersect_cell_pt1(:,2) = cell_minval(:,2);
            temp = [cell_minval(2:size(Intersect_cell_pt1,1),2);1];
            Intersect_cell_pt1(:,3) = temp;
        end
        
        %% climb down one by one, until gets to the minimum bound
        %% (when ind becomes empty)
        ind = 0; maxval = xb(i);
        vk = PTS(index,:); % the node of the voronoi cell in which xb resides
        
        while ~isempty(ind)
            x = []; % the coordinates of the intersection
            
            %% calculate the intersection points
            %% using Sambridge (1999) eq.19,24
            dk = sum((vk(1:NDIM)-xb(1:NDIM)).^2) - (vk(i)-xb(i)).^2;
            for j = 1:size(PTS,1)
                if vk(i) == PTS(j,i)
                    x(j) = NaN;
                else
                    % Sambridge (1999a) eq. 19, all the intersection points
                    x(j) = 1/2 .* (vk(i)+PTS(j,i)+((dk-dj(j))./(vk(i)-PTS(j,i))));
                end
            end
            
            % retrieve the cell number which is next to the
            % previous one and the intersection point
            ind = find(0 < x & x < maxval);
            if ~isempty(ind)
                [maxval, Ind] = max(x(ind));
                cell_maxval = [ind(Ind), maxval; cell_maxval];
                vk = PTS(ind(Ind),:);
            end
        end
        
        %% Save djprev, iprev for the next (i+1th) axis
        iprev = i;
        djprev = dj;
        
        % change Intersect_cell_pt1 in such a way that it
        % contains the intervals
        if ~isempty(cell_maxval)
            Intersect_cell_pt2(:,1) = cell_maxval(:,1);
            Intersect_cell_pt2(:,3) = cell_maxval(:,2);
            temp = [0;cell_maxval(1:size(Intersect_cell_pt2,1)-1,2)];
            Intersect_cell_pt2(:,2) = temp;
        end
        
        % Include the starting cell
        if (isempty(Intersect_cell_pt1) && isempty(Intersect_cell_pt2))
            Intersect_cell_pt = [index, 0, 1];
        elseif isempty(Intersect_cell_pt2)
            Intersect_cell_pt = [ ...
                index, 0, Intersect_cell_pt1(1,2); ...
                Intersect_cell_pt1];
        elseif isempty(Intersect_cell_pt1)
            Intersect_cell_pt = [Intersect_cell_pt2; ...
                index, Intersect_cell_pt2(size(Intersect_cell_pt2,1),3), 1];
        else
            Intersect_cell_pt = [Intersect_cell_pt2; ...
                index, Intersect_cell_pt2(size(Intersect_cell_pt2,1),3), Intersect_cell_pt1(1,2); ...
                Intersect_cell_pt1];
        end
        
        %% Put a random deviate along i-th axis which has a
        %% probability distribution defined by Intersect_cell_pt
        %% using 'transformation method'
        
        %% change misfit to normalized probability density %%
        % minmisfit is the minimum misfit along the current axis
        misfit = []; foo = []; probdens = [];
        misfit = PTS(Intersect_cell_pt(:,1),NDIM+1);
        minmisfit = min(misfit);
        % probdens at minmisfit is 1
        foo = (-PTS(Intersect_cell_pt(:,1),NDIM+1)+minmisfit)./2;
        probdens = exp(foo);
        
        % interval between cell boundaries
        intv = [];
        intv = Intersect_cell_pt(:,3)-Intersect_cell_pt(:,2);
        
        % probability = lenght of cell along dimension * probdens
        % different prob along different directions
        prob = [];
        prob = intv.*probdens;
        
        % cumulate probability along the axis so that the sum is one
        cumul_prob = [];
        cumul_prob(1) = prob(1);
        for kk = 2:size(prob,1)
            cumul_prob(kk) = cumul_prob(kk-1) + prob(kk);
        end
        
        % normalize cumulative probability
        ncumul_prob = [];
        ncumul_prob = cumul_prob./cumul_prob(length(cumul_prob));
        
        % random seed
        rand('state',sum(100*clock));
        seed = rand;
        
        % Which interval of cumul prob or dimension of the voronoi cell does seed belong to?
        % The higher the probdens (intv*ppd) along a direction, the more likely will
        % the deviate be in this direction
        match = [];
        if seed < ncumul_prob(1)
            match = 1;
        else
            for kk = 2:length(ncumul_prob)
                if ncumul_prob(kk-1) < seed && seed < ncumul_prob(kk)
                    match = [match kk];
                end
            end
            match=min(match);
        end
        if isempty(match)
            disp('something is wrong in putting a deviate!');
            %return
        end
        
        % put a deviate in the 'match'-th direction;
        % deviate is proportional to cell length along match dimension
        % and to the seed nb
        xb(i) = rand.* ...
            (Intersect_cell_pt(match,3)-Intersect_cell_pt(match,2)) ...
            + Intersect_cell_pt(match,2);
        
        % for the next (i+1th) axis
        index = Intersect_cell_pt(match,1);
        
        
        %%%%% temporarily put %%%%%
        
        %         disp(['Intersect_cell_pt1 = ']);
        %         disp([num2str(Intersect_cell_pt1)]);
        %         disp(['Intersect_cell_pt2 = ']);
        %         disp([num2str(Intersect_cell_pt2)]);
        %         disp(['Intersect_cell_pt = ']);
        %         disp([num2str(Intersect_cell_pt)]);
        %         disp(['i = ' num2str(i) ' index = ' num2str(index)]);
        
        %%%%%%%%%%%%%%%%%%%%%
        
    end
    
    %% dimensionalize
    xbn = [];
    for comp = 1:NDIM
        xbn(:,comp) = xb(:,comp).*(maxlim(comp)-minlim(comp))+minlim(comp);
    end
    
    outpts = [outpts; xbn(1:NDIM), misfit(match)];
    
    
end
end

