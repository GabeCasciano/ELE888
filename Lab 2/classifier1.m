function [result] = classifier1(weight, feature)
%CLASSIFIER Summary of this function goes here
%   Detailed explanation goes here
    g = discriminant(weight, feature);
    g = g - weight(1,1)*feature(1,1);
    if(g > weight(1,1))
        result = 1;
    elseif (g == weight(1,1))
        result = 0;
    else
        result = 2;
end

