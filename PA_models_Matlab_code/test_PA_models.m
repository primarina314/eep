clear
close all
clc

% Script for testing the function PAmodel().
% 
% By Lauri Anttila and Mahmoud Abdelaziz, Tampere University of Technology, 
% 2018-04-04

%% Define simulation parameters and load PA models
backoff = 0;  % Back-off (in dB's) from the maximum TX power (i.e., the power 
              % at which the models were identified)
              % - negative values may be used, but with caution (always check 
              %   AM/AM response)
gains = 'measured'; % 'equal' or 'measured'; use equal (unit) gains, or measured (normalized around unity) gains for the transmitters

% Load PA memory models
load('PA_memory_models');  % variable name is 'parameters'

if strcmpi(gains,'equal')
    Gain_tx = ones(100,1);
else
    % Load measured TX powers, and map them to amplitude gains, normalized around 1 
    load('P_TX_Case1.mat')  % variable name is P_tx
    Gain_tx = sqrt(10.^((P_tx)/10))/mean(sqrt(10.^((P_tx)/10)));
end

%% Generate test signal (rectangular QAM with RRC filtering, as an example)
Nsym = 30000; % Number of QAM symbols used in the simulation
osf = 8;      % Oversampling factor
M = 256;      % QAM alphabet size
tmp = -(sqrt(M)-1):2:(sqrt(M)-1);
alphabet = kron(tmp,ones(1,sqrt(M))) + 1i*kron(ones(1,sqrt(M)),tmp);
index = ceil(M*rand(Nsym,1));
sym = alphabet(index);
input = pulse_shape(sym,osf,0.22,12);
Fs_in = 120e6;  % sample rate of the generated signal

%% PA models
Num_PAs = 16; % Number of different PA models used in this simulation (max 100)
rf  = 0;      % 0 or 1 (referring to TX0 or TX1 in a given USRP)
for Ind = 1:Num_PAs % index of the USRP to be analyzed; from 1 to 50
    indTX = 2*(Ind)-1+rf; % index of transmitter (from 1 to 100)

    param = parameters{indTX};
    
    if strcmpi(gains,'measured') % use measured gains, normalized around 1
        param.Gain = Gain_tx(indTX);
    else % equal gains
        param.Gain = 1; % all TX's have same gain
    end

    % PA model
    output = PAmodel(input,param,backoff);

    % Plot PSD's of PA outputs
    PSD_out = pwelch(output,kaiser(2048,7));
    figure(1); hold on;
    plot(Fs_in/2*(-1:2/(2^11):1-2/(2^11)),10*log10(abs(fftshift(PSD_out))))
    xlabel('Frequency [Hz]')
    ylabel('PSD [dB]')
    % Plot AM/AM responses of the PA models:
    figure(2); hold on;
	plot(abs(input),abs(output),'.')
    xlabel('Input amplitude')
    ylabel('Output amplitude')
    
end
