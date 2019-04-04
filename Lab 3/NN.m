classdef NN
    properties
        numberOfHidden
        numberOfInputs
        numberOfOutputs
        hiddenLayer = nueron;
        outputLayer = nueron;
        bias;
    end
    
    methods
        function obj = NN(numberOfHidden, numberOfInputs, numberOfOutputs)
            obj.numberOfHidden = numberOfHidden;
            obj.numberOfInputs = numberOfInputs;
            obj.numberOfOutputs = numberOfOutputs;
            obj.bias = zeros(2, numberOfHidden);
            obj.hiddenLayer(numberOfHidden) = nueron(numberOfInputs);
            obj.outputLayer(numberOfOutputs) = nueron(numberOfOutputs);
            
            for i = 1 : numberOfHidden
                obj.hiddenLayer(i) = nueron(numberOfInputs);
            end
            for i = 1 : numberOfOutputs
                obj.outputLayer(i) = nueron(numberOfOutputs);
            end
        end
        
        function [z, y] = calculateOutput (obj, input)
            z = zeros(1, obj.numberOfOutputs);
            y = zeros(1, obj.numberOfHidden);
            
            for j = 1 : obj.numberOfHidden
                y(j) = obj.hiddenLayer(j).activation([input obj.bias(1,j)]);
            end
            for j = 1 : obj.numberOfOutputs
                z(j) = obj.hiddenLayer(j).activation([y obj.bias(2,j)]);
            end
        end
        function [Wj, Wk] = backPropigate(obj, eta, theta, inputMatrix, target, iter)
            missClassedInput = zeros(2, length(inputMatrix));
            missClassedInput = inputMatrix
            prevNetk = zeros(1, obj.numberOfOutputs);
            prevWk = zeros(1, obj.numberOfOutputs);
            deltak = zeros(1, obj.numberOfOutputs);
            deltaWk = zeros(1, obj.numberOfOutputs);
            deltaj = zeros(1, obj.numberOfHidden);
            deltaWj = zeros(1, obj.numberOfHidden);
            sumWk = zeros(1, obj.numberOfHidden);
            for e = 1 : inter
                for i = 1 : length(inputMatrix)
                    for k = 1 : obj.numberOfOutputs
                        deltak(k) = (target(k) - obj.outputLayer(k).activation(inputMatrix(i,:))) * obj.outputLayer(k).deltaActivation(inputMatrix(i,:));
                        deltaWk(k) = deltaWk(k) + eta * deltak(k) * (prevNetk(k) - obj.outputLayer(k).discriminant([1 inputMatrix(i, :)]))/(prevWk(k) - Wk(k));
                        prevNetk(k) = obj.outputLayer(k).discriminant([1 inputMatrix(i, :)]);
                        prevWk(k) = 
                        obj.outputLayer(k).weightVector = obj.outputLayer(k).weightVector + deltaWk;
                        sumWk = sumWk + (obj.outputLayer(k).weightVector + deltak);
                        obj.outputLayer
                    end
                end
                for i = 1 : length(inputMatrix)
                    for j = 1 : obj.numberOfHidden
                        deltaj(j) = obj.hiddenLayer(j).deltaActivation(inputMatrix(i, :)) * sumWk;
                        deltaWj(j) = deltaWj(j) + (eta * deltaj(j) * inputMatrix(i,:));
                        
                        Wj(j) = Wj(j) + deltaWj(j);
                    end
                end
            end
        end
    end
end

