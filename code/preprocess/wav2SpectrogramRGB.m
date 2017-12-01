function [ spectroEnergyFlat ] = wav2SpectrogramRGB( recording, Fs, width, height )
% convert waveform to spectrogram
logSign = 1;

if nargin <= 2
    width = 150;
    height = 64;
end

spectro = spectrogram( recording, floor( size( recording, 2 )/( width /2 + 1) ) , floor( size( recording, 2 )/ ( width + 1 ) ) - 1, 2*height , Fs, 'yaxis');
%spectrogram( recording, floor( size( recording, 2 )/( width /2 + 1) ) , floor( size( recording, 2 )/ ( width + 1 ) ) - 1, 2*height , Fs, 'yaxis');

%colormap gray

%imshow( abs(spectro), [ min( min( abs( spectro) ) ), max( max( abs(spectro) ) ) ] );
%imshow( abs(spectro) );

if logSign == 1
   spectroEnergy =  log( abs( spectro ).^2 + 0.00000000001 );
else
   spectroEnergy =  abs( spectro ).^2 ;
end

%imshow( spectroEnergy, [ min( min( spectroEnergy ) ), max( max( spectroEnergy ) ) ] )
%imshow( spectroEnergy )

% rotate the spectrogram to correct position
spectroEnergy = flipdim( spectroEnergy, 1 );

%## do not conduct normalization on a single spectrogram
%spectroEnergy = spectroEnergy/ ( max( max ( spectroEnergy ) ) - min( min ( spectroEnergy ) ) ) ;

% cut the edge
spectroEnergy = spectroEnergy( 1 : height, 1: width ); 

%imshow( spectroEnergy, [ min( min( spectroEnergy ) ), max( max( spectroEnergy ) ) ] );

%% flatten to one dimension, easier for further processing
%NOTICE: RESHAPE WILL ROTATE THE ORIGNAL FIGURE
spectroEnergyFlat = reshape( spectroEnergy', [ 1, height *width ] );

%imshow( spectroEnergyFlat, [ min( min( spectroEnergyFlat ) ), max( max( spectroEnergyFlat ) ) ] )

end

