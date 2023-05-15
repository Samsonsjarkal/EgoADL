function [y, Fs] = sound_feature(filename)
    [y,Fs] = audioread(filename);
    return