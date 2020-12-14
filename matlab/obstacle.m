function [xx,uu] = obstacle(m,sweeps,plotsweeps)
% Solve a 1D obstacle problem:          /1
%                                min    |  1/2 (u')^2 - f u
%                             u >= phi  /0
% where  phi(x) = sin(3 pi x) - 1/2  and  f(x) = -1
% FIXME this version is projected Gauss-Seidel

if nargin < 3,  plotsweeps = false;  end
if nargin < 2,  sweeps = 3;  end
if nargin < 1,  m=8;  end
f = @(x) -ones(size(x));
phi = @(x) sin(3*pi*x) - 0.5;

% mesh
xx = linspace(0.0,1.0,m+1);  h = 1.0 / m;
ff = f(xx);  pphi = phi(xx);

% initial iterate is feasible
uu = max(phi(xx),zeros(size(xx)));
if plotsweeps,  stdplot('initial',xx,uu),  end

% sweeps of projected GS
for s = 1:sweeps
    for l = 2:m
        c = (h/2) * ff(l) + (uu(l-1) + uu(l+1)) / 2 - uu(l);
        uu(l) = uu(l) + max(c,pphi(l)-uu(l));
    end
    if plotsweeps,  stdplot(sprintf('sweep %d',s),xx,uu);  end
    fprintf('.')
end
fprintf('\n')

    function stdplot(str,xx,uu)
        figure(1), clf
        pp = xx(2:end-1);
        plot(xx,uu,xx,phi(xx),'r',...
             pp,zeros(size(pp)),'ko','markersize',8)
        xlabel x,  axis tight,  title(str)
        %pause(1)
    end
end % function obstacle

