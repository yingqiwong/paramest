% Some examples of Gibbs sampling, resampling, annealing
% Sarah Minson, May 4, 2015
clear all

if 1
disp('Gibbs sampler')
% Gibbs sampling a bivariate normal distribution
% Gibbs is popular because candidate samples are always accepted
% From https://theclevermachine.wordpress.com/2012/11/05/mcmc-the-gibbs-sampler/
% EXAMPLE: GIBBS SAMPLER FOR BIVARIATE NORMAL
rand('seed' ,12345);
nSamples = 5000;
 
mu = [0 0]; % TARGET MEAN
rho(1) = 0.8; % rho_21
rho(2) = 0.8; % rho_12
 
% INITIALIZE THE GIBBS SAMPLER
propSigma = 1; % PROPOSAL VARIANCE
minn = [-3 -3];
maxx = [3 3];
 
% INITIALIZE SAMPLES
x = zeros(nSamples,2);
x(1,1) = unifrnd(minn(1), maxx(1));
x(1,2) = unifrnd(minn(2), maxx(2));
 
dims = 1:2; % INDEX INTO EACH DIMENSION
 
% RUN GIBBS SAMPLER
t = 1;
while t < nSamples
    t = t + 1;
    T = [t-1,t];
    for iD = 1:2 % LOOP OVER DIMENSIONS
        % UPDATE SAMPLES
        nIx = dims~=iD; % *NOT* THE CURRENT DIMENSION
        % CONDITIONAL MEAN
        muCond = mu(iD) + rho(iD)*(x(T(iD),nIx)-mu(nIx));
        % CONDITIONAL VARIANCE
        varCond = sqrt(1-rho(iD)^2);
        % DRAW FROM CONDITIONAL
        x(t,iD) = normrnd(muCond,varCond);
    end
end
 
% DISPLAY SAMPLING DYNAMICS
figure(1); clf; subplot(2,1,1);
h1 = scatter(x(:,1),x(:,2),'r.');
 
% CONDITIONAL STEPS/SAMPLES
hold on;
for t = 1:50
    plot([x(t,1),x(t+1,1)],[x(t,2),x(t,2)],'k-');
    plot([x(t+1,1),x(t+1,1)],[x(t,2),x(t+1,2)],'k-');
    h2 = plot(x(t+1,1),x(t+1,2),'ko');
end
 
h3 = scatter(x(1,1),x(1,2),'go','Linewidth',3);
legend([h1,h2,h3],{'Samples','1st 50 Samples','x(t=0)'},'Location','Northwest')
hold off;
xlabel('x_1');
ylabel('x_2');
axis square
title('Gibbs sampling')

subplot(2,1,2);
X=mvnrnd([0 0]',[1 rho(2); rho(1) 1],nSamples);
scatter(X(:,1),X(:,2),'r.');
axis square
title('Direct sampling');
end

% Gibbs sampling only works if the joint PDF can be uniquely described
% by its conditional PDFs.
% A problem invented to cause the Gibbs sampler to fail from
% O'Hagan and Forster section 10.25
% Consider a bivariate distribution that's equal to constant c in the
% region 0 <= theta1, theta2 <= 1, and 1-c on the region 2 <=
% theta1,theta2, <= 3.  (0<c<1)
N=1e4;
c=0.5;
figure(2); clf; nyp=3; nxp=1;
subplot(nyp,nxp,1);
xv=[0 1 1 0]';
yv=[0 0 1 1]';
patch([xv xv+2],[yv yv+2],[c 1-c]);
set(gca,'dataaspectratio',[1 1 1],'xlim',[0 3],'ylim',[0 3]);
title('Target PDF')

for mycase=1:2
    switch mycase
        case 1 % Case 1: 0 <= theta0(2) <= 1
            theta0_2=.5;
        case 2 % Case 2: 2 <= theta0(2) <= 3
            theta0_2=2.5;
    end
    X=nan*ones(2,N);
    for k=1:2*N
        if k==1
            theta0=theta0_2;
        else
            theta0=X(k-1);
        end
        if theta0<=1
            X(k)=unifrnd(0,1);
        else
            X(k)=unifrnd(2,3);
        end
    end
    subplot(nyp,nxp,mycase+1); hold on
    scatter(X(1,:)',X(2,:)','.r');
    set(gca,'dataaspectratio',[1 1 1],'xlim',[0 3],'ylim',[0 3]);
    title(['Starting location \theta_2=' num2str(theta0_2)]);
end
suptitle('How to break the Gibbs sampler')

% Resampling - Only have to calculate things once
target=@(x)normpdf(x,0,1); % Target PDF ~ N(0,1)
X=unifrnd(-10,10,1,1e4); % Generate initial samples from some other PDF
w=target(X); % Assign weights to each sample based on how probable they are in target PDF
w=w/sum(w); % Normalize weights to sum to 1
% Here's the trick: draw random samples with probability according w
% If we draw random numbers between 0 and 1 and histogram into bins whose
% width is proportional to w, then the frequency in each bin will be
% proportional to w.
xc=[0 cumsum(w)]; % bins extend from zero to 1, width of bin_i is proportional to w_i
count=histc(rand(size(X)),xc);
count(end-1)=count(end-1)+count(end); count=count(1:end-1);
% Now change frequency of original samples according to frequency in
% corresponding bins
ind=[]; for i=1:length(count); ind=[ind repmat(i,1,count(i))]; end
X2=X(ind);

figure(3); clf; subplot(1,1,1); hold on
x=-10:.1:10;
plot(x,target(x),'-k');
n=hist(X2,x);
n=n/sum(n)/diff(x(1:2));
plot(x,n,'-r');
legend('Target','Resampling')
title([num2str(length(unique(X2))) ' unique samples from ' num2str(length(X)) ' samples']);

if 1
disp('Parallel Tempered MCMC')
% Annealing example:
% Parallel Tempered MCMC
% Run multiple MCMC chains at different temperatures
% http://www.cs.ubc.ca/~nando/540b-2011/projects/8.pdf
% https://darrenjw.wordpress.com/2013/09/29/parallel-tempering-and-metropolis-coupled-mcmc/


% go from likelihood^0 to likelihood ^1

U = @(x)(x.^2-1).^2;
target = @(x)exp(-U(x));
p = @(x,tk)exp(-U(x)*tk);
beta=exp(linspace(log(0.1),log(1),5));
T=1./beta;
T=2.^[0:3];
M=length(T); % Number of temperatures
Niter=1e4; % Number of steps in each Metropolis chain
Nsweep=1; % Number of sweeps through the PTMCMC algorithm
X=nan*ones(M,Niter);
X(:,1)=-1;
for j=2:size(X,2) % Run Metropolis
    for k=1:M
        tk=T(k);
        x=X(k,j-1);
        y=x+normrnd(0,0.5);
        px=p(x,tk);
        py=p(y,tk);
        alpha=py/px;
        u=rand;
        if u<=alpha
            X(k,j)=y; % Accept candidate
        else
            X(k,j)=x; % Reject candidate
        end
    end
    n=histc(unifrnd(0,M,[1 2]),0:M);
    n(end-1)=n(end-1)+n(end); n=n(1:end-1);
    kk=find(n);
    if length(kk)==1; kk=[kk;kk]; end
    tk=T(kk(1));
    tk1=T(kk(2));
    xk=X(kk(1),j);
    xk1=X(kk(2),j);
    % Consider whether to swap samples
    alpha=p(xk1,tk)*p(xk,tk1)/[p(xk,tk)*p(xk1,tk1)];
    alpha=min([1 alpha]);
    u=rand;
    if u<=alpha; % swap samples
        X(kk,j)=X(flipud(kk),j);
    end
end
Xpt=X;

N=M*Niter; % Same number of samples as PT-MCMC
% Compare to regular MCMC
X=nan*ones(1,N);
X(:,1)=-1;
Nacc=0;
 for i=1:size(X,1) % For each proposal distribution q~N(0,qsigma)
    for j=2:size(X,2) % Run Metropolis
        x=X(i,j-1);
        y=x+normrnd(0,0.5);
        alpha=target(y)/target(x);
        u=rand;
        if u<=alpha
            X(i,j)=y; % Accept candidate
            Nacc(i)=Nacc(i)+1;
        else
            X(i,j)=x; % Reject candidate
        end
    end
 end
figure(4); clf; subplot(1,1,1); hold on;
xc=-2:.05:2;
[n,x]=hist(X(:,201:end)',xc); % Discard first 200 samples as burn-in
n=n./repmat(sum(n),size(n,1),1)/diff(x(1:2));
ii=find(T==1); myX=Xpt(i,1:end);
[npt,x]=hist(myX(:),xc); % Discard first 200 samples as burn-in
npt=npt./sum(npt)/diff(x(1:2));
y=target(x); y=y/trapz(x,y);
plot(x,y,'-k','linewidth',5)
plot(x,n,'-b','linewidth',2);
plot(x,npt,'-r','linewidth',2);
legend('Target','Metropolis','Parallel Tempered MCMC');
end

clear all
disp('CATMIP'); tic
% Finally, because we can, CATMIP
% Transitioning (special case of tempering) + resampling + Metropolis
target=@(x)0.9*mvnpdf(x,-0.5*ones(size(x)),0.01*eye(length(x)))+0.1*mvnpdf(x,0.5*ones(size(x)),0.01*eye(length(x))); % Target PDF is weighted sum of Gaussians

N=5e3; Nsteps=5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initializaton
Cmsqr   = 0.1*0.1;      % Scale factor for proposal PDF q(Cmsqr*Sm)
beta    = 0.0;          % cooling temperature
dbeta   = 5.e-5;        % initial guess for change in beta
beta = 0; c = 0;
m=0;

Nparam=2;

% Directly sample the prior distribution
theta=unifrnd(-2,2,[Nparam N]);
LLK=ones(1,N); for i=1:N; LLK(i)=log(target(theta(:,i))); end

fprintf('m\tCm^2\tCOV\tbeta\t\tNaccept\t\tNreject\n');
fprintf('%d\t%.4f\t%.4f\t%.4e\t%.4e\t%.4e\n',m,Cmsqr,c,beta,0,0);

while 1
  m=m+1;
  
  % Update temperature
  % Most efficient sampling achieved when new beta is chosen so that 
  % cov(w)=1 (Beck and Zuev, 2013)
  % User a helper function to do a numerical solve for beta s.t. cov(w)=1
  [w,beta,dbeta,c]=catmip_calc_beta(LLK,beta,dbeta);
  fprintf('%d\t%.4f\t%.4f\t%.4e\t',m,Cmsqr,c,beta);
  
  % Resample to match new target PDF
  count=histc(rand([1 N]),[0 cumsum(w)]);
  count(end-1)=sum(count(end-1:end));
  count=count(1:end-1);
  ind=[]; for i=1:length(count); ind=[ind repmat(i,1,count(i))]; end
  theta=theta(:,ind);
  
  % ...But now we've lost diversity in our sample population
  % So let's update each of our samples with a Metropolis chain, replacing
  % each current sample with the last one from its chain.  Now we hopefully
  % have regained our diversity, and explored new and interesting parts of
  % the model space that our previous samples hadn't visited
  
  % Use a proposal density that is a Gaussian approximation of the target,
  % N(0,Sm), and rescaled according to the rejection rate
  %PROCEDURE:
  %1. Calculate p=w/sum(w)
  %2. Calculate the expected value: E = sum(p_i*theta_i)
  %3. Calcluate Sm = sum{p_i*theta_i*theta_i^T} - E*E^T
  %4. Return Cm^2 * Sm
  p=w/sum(w);
  E=sum([repmat(p,size(theta,1),1).*theta],2);
  Sm=zeros(size(theta,1));
  for i=1:size(theta,2); Sm=Sm+p(i)*theta(:,i)*theta(:,i)'; end
  Sm=Sm-E*E';
  Sm=Cmsqr*Sm;
  Sm = 0.5*(Sm + Sm'); % Make sure that Sm is symmetric
  
  % Run Nsteps of Metropolis on each sample
  IOacc=zeros(Nsteps-1,N);
  parfor ii=1:N % Loop over samples: each sample is the seed for a Markov chain
      X=zeros(Nparam,Nsteps);
      X(:,1)=theta(:,ii); % Our current sample is the seed for the chain
      Xllk=zeros(1,Nsteps);
      Xllk(1)=log(target(theta(:,ii)));
      z=mvnrnd(zeros(Nparam,1),Sm,Nsteps)';
      for k=1:Nsteps-1 % Run Metropolis
          x=X(:,k);
          y=X(:,k) + z(:,k);
          px=Xllk(k);
          py=log(target(y));
          r = beta*(py-px);
          u=log(rand);
          if u<=r
              X(:,k+1)=y; Xllk(:,k+1)=py; IOacc(k,ii)=1; %Naccept=Naccept+1;
          else
              X(:,k+1)=x; Xllk(:,k+1)=px; %Nreject=Nreject+1;
          end
      end
      theta(:,ii)=X(:,end); % Save only the last sample from the Markov chain
      LLK(:,ii)=Xllk(:,end); % Now the original sample has been updated by MCMC
  end
  Naccept=sum(IOacc(:));
  Nreject=length(IOacc(:))-Naccept;
      
  fprintf('%.4e\t%.4e\n',Naccept,Nreject);
  
  % Rescale step size by acceptance rate per Matt Muto
  accRatio = Naccept/(Naccept + Nreject);
  kc = (8*accRatio + 1)/9;
  % //kc = max(kc,0.2);   kc = min(kc,1);
  if (kc < 0.001); kc = 0.001; end
  if (kc > 1.0); kc = 1.0; end
  Cmsqr = kc * kc;
  
  if (1-beta < 0.005); fprintf('mstop=%d\n',m); mstop=m; break; end
end
toc

figure(5); clf; nyp=2; nxp=2;
dx=0.05; xl=[-1 1];
bins=xl(1)-dx:dx:xl(2)+dx;
ctrs={bins bins};
[x,y]=meshgrid(bins,bins);
z=nan*x; for i=1:length(x(:)); z(i)=target([x(i) y(i)]); end
n=hist3(theta',ctrs);
n=n/sum(n(:))/dx/dx;
subplot(nyp,nxp,1); surf(x,y,z); xlim(xl); ylim(xl); title('Target')
subplot(nyp,nxp,2); surf(x,y,n); xlim(xl); ylim(xl); title(['CATMIP (' num2str(N) ' samples)'])
n=hist(theta',bins);
n=n./repmat(sum(n),length(bins),1)/dx;
z=nan*n(:,1); for i=1:length(bins); z(i)=target(bins(i)); end
subplot(nyp,nxp,3); hold on; title('Marginal PDFs')
plot(bins,z,'-k','linewidth',5)
h=plot(bins,n,'linewidth',3); set(h(1),'color','b'); set(h(2),'color','r','linestyle','--');
leg={'Target';'CATMIP dim1';'CATMIP dim2'}; xlim([-1 1]);

% Compare to Metropolis
disp('Use Metropolis to run same problem for same number of samples'); tic
Ntotal=mstop*N*(Nsteps-1) + N % Total number of model evaluations by CATMIP
X=zeros(Nparam,Ntotal+1); X(:,1)=[-1 -1]'; Xllk=zeros(1,Ntotal+1); Xllk(1)=log(target(X(:,1)));
z=mvnrnd(zeros(Nparam,1),0.01*eye(Nparam),Ntotal)';
for k=1:Ntotal % Run Metropolis
    x=X(:,k);
    y=X(:,k) + z(:,k);
    px=Xllk(k);
    py=log(target(y));
    r = beta*(py-px);
    u=log(rand);
    if u<=r
        X(:,k+1)=y; Xllk(:,k+1)=py;
    else
        X(:,k+1)=x; Xllk(:,k+1)=px;
    end
end; toc
X=X(:,202:end); Xllk=Xllk(:,202:end); % Discard burn-in
n=hist3(X',ctrs); [x,y]=meshgrid(bins,bins);
n=n/sum(n(:))/dx/dx;
subplot(nyp,nxp,4); surf(x,y,n); xlim(xl); ylim(xl); title(['Metropolis (' num2str(size(X,2)) ' samples)'])
n=hist(X',bins);
n=n./repmat(sum(n),length(bins),1)/dx;
z=nan*n(:,1); for i=1:length(bins); z(i)=target(bins(i)); end
subplot(nyp,nxp,3); hold on
h=plot(bins,n,'linewidth',3);  set(h(1),'color','c'); set(h(2),'color','y','linestyle','--'); leg(end+1)={'Metro dim1'}; leg(end+1)={'Metro dim2'};

legend(leg); for ip=[1 2 4]; subplot(nyp,nxp,ip); xlabel('dim 1'); ylabel('dim 2'); end
return

% exponent of likelihood, beta, is an exponential function of the number of
% beta steps.  Integrating expected value of likelihood^beta wrt beta gives
% the integral of P(d|m)P(m) dm i.e. the denominator in Bayes
% (Thermodynamic integral estimator (TIE))
% choosing a beta gives you a set of seeds that are almost the posterior - 
% do random walk on these seeds to do MCMC (number of parallel walks is
% equal to no of seeds) so you end up getting the exact posterior distribution.  



