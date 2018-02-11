function K = covESAM( hyp, x, z, i)

% Polynomial covariance function. The covariance function is parameterized as:
%
% k(x^p,x^q) = sf^2 * ( c + (x^p)'*(x^q) )^d 
%
% The hyperparameters are:
%
% hyp = [ log(c)
%         log(sf)  ]
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.
%
% See also COVFUNCTIONS.M.

if nargin<2, K = '2'; return; end                  % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

sf0 = exp(hyp(1));                                          % inhomogeneous offset
sf1 = exp(hyp(2));                                           % signal variance

% precompute inner products
if dg                                                               % vector kxx
  K = (sf0*sf0)*ones(size(x,1),1);
else
  if xeqz                                                 % symmetric matrix Kxx
    xx = x*x';
    xnorm = sqrt(sum( x.^2, 2) );
    K = xx ./ (xnorm*xnorm'+1e-10);
  else                                                   % cross covariances Kxz
    xz = x*z';
    xnorm = sqrt(sum( x.^2,2) );
    znorm = sqrt(sum( z.^2,2) );

    K = xz ./ (xnorm*znorm'+1e-10);
  end
end
K = max( K, -1 );
K = min( K, 1 );

K = acos(K);    

if nargin<4                                                        % covariances
  K = (sf0*sf0)* exp(-K/(sf1*sf1));
else                                                               % derivatives
  if i==1
    K = 2*sf0*exp(-K/(sf1*sf1));
  elseif i==2
    K = 2*(sf0*sf0)/(sf1^3)*K.*exp(-K/(sf1*sf1));
  else
    error('Unknown hyperparameter')
  end
end
