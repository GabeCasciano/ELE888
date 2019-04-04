classdef nueron
    %NUERON Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        weightVector
        numberOfInputs
    end
    
    methods
        function obj = nueron(numberOfInputs)%constructor
            if nargin > 0
                obj.weightVector = zeros(1, numberOfInputs + 1);
                obj.numberOfInputs = numberOfInputs;
            end
        end
        function g = discriminant(obj, input)%calculate discriminant
            g = 0;
            for i = 1 : obj.numberOfInputs + 1
                g = g + (obj.weightVector(i) * input(i));
            end
        end
        function y = activation(obj, input)
            y = tanh(discriminant(obj, [1 input]));
        end
        function z = deltaActivation(obj, input)
            z = sech(discriminant(obj, [1 input]));
        end
        
    end
end

