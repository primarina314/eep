function [s,d,rrc] = pulse_shape(sym,over,alpha,D)

% Inputs:
% * sym - symbol sequence
% * over - oversampling factor
% * alpha - roll-off of the root raised cosine (RRC) pulse (0<alpha<=1)
% * D - delay of the RC filter in symbol periods

sym = sym(:);
sym_over = kron(sym,[1; zeros(over-1,1)]);

rrc = rcosdesign(alpha,D*2,over,'sqrt');
[~,d] = max(rrc);

s = conv(rrc,sym_over);
s = s(d:d+length(sym)*over-1);
