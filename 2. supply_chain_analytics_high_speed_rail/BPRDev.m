function [t,devt]= BPRDev(f,t0,a,b,c)
% the derivative of bpr function 
%a=50;
t=t0*(1+a*(f/c)^b);
devt=t0*a*b*((f/c)^(b-1));
end