function [result] = classifier3(weight, feature)
%CLASSIFIER Summary of this function goes here
%   Detailed explanation goes here
    g = discriminant(weight, feature);
    g = g - weight(1,1)*feature(1,1);
    if(g > weight(1,1))
        result = 2;
    elseif (g == weight(1,1))
        result = 0;
    else
        result = 3;
end

