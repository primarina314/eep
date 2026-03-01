function [output] = PAmodel(input,param,backoff) 

% Power amplifier model. Models extracted from the Lund University massive
% MIMO testbed at 2 GHz carrier frequency, and at 120 Msps sample rate.
% 
% Inputs:
% - input = PA input signal
% - param = struct containing the PA parameters, with fields
%   ** param.pa.P = polynomial order
%   ** param.pa.Lpa = vector containing the memory length per each (odd) polynomial order
%   ** param.pa.coeff = polynomial coefficients
%   ** param.Gain = amplitude gain of the PA model
% - backoff = power back-off in dB. 0 dB back-off corresponds to input
%   power=1, with which the PA models were identified
% 
% By Lauri Anttila and Mahmoud Abdelaziz, Tampere University of Technology, 
% 2018-04-04


%% Scale input power according to the specified back-off  
Pin = mean(abs(input).^2);  % store the input power, and scale back to it after the TX model
scale_input = 1/sqrt(10^(backoff/10)*Pin); % 0 dB back-off corresponds to input power=1
input = scale_input*input;

%% Memory polynomial PA model
L_in = length(input);
M = param.pa.Lpa; % memory order of the PA
P = param.pa.P; % nonlinearity order of the PA

% Generate static basis functions
PHI = zeros(L_in,(P+1)/2);
for p=1:(P+1)/2
    PHI(:,p) = input.*abs(input).^(2*(p-1));
end

% Generate memory terms
R = zeros(L_in+max(M)-1,sum(M));
for pp = 1:(P+1)/2
    for m = 1:M(pp)
        R(m:L_in+m-1,sum(M(1:pp-1))+m) = PHI(1:L_in,pp);
    end
end

output = R(1:L_in,:)*param.pa.coeff;

%% TX specific gain (unity gain on average)
output = param.Gain*output;

%% Scale power back to original
output = 1/scale_input*output;

end
