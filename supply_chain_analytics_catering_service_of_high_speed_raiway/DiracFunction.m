function [Dirac,DiracDev] = DiracFunction(x)
% Construc a dirac delta function bassed on Gauss distribution 
%normpdf(x,0,SIGMA)=Gauss dense probility function

SIGMA=1e-3;
Dirac= normpdf(x,0,SIGMA);
DiracDev=((-2.*x)./SIGMA).*normpdf(x,0,SIGMA);


end

