function [x1, x2] = decisionBoundary(weight)
%DECISIONBOUNDARY Summary of this function goes here
%   Detailed explanation goes here
    x1 = weight(1,1) / weight(1,2);
    x2 = weight(1,1) / weight(1,3);
end


