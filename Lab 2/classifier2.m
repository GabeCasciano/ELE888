function [result] = classifier2(weight, feature)
%CLASSIFIER Summary of this function goes here
%   Detailed explanation goes here
    g = discriminant(weight, feature);
    if(g > 0)
        result = 1;
    elseif (g == 0)
        result = 0;
    else
        result = 2;
end

