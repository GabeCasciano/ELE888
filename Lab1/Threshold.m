%determining the treshold for x1 & x2

%error_x1 is used to calculate the error for threshold_x1
%error_x2 & threshold_x2 are the same as their x1 counter parts
%% Variables
% These are the variables that will be used to classify the flower based on
% the output discriminant function .

error_x1 = 0;
threshold_x1 = 0;
error_x2 = 0;
threshold_x2 = 0;
%% Calculations
% This loop is being used to calculate the disctiminat function for all of
% the data set in the training set. The training set is being used because
% it has datalabels associated with each data set, thus making it easier to
% determine whether or not the correct decision has been made.

for i = 1 : length(trainingSet)
    
    %the discriminant and posteriror probabilities for feature 1 & 2 are
    %being calculated from the lab1_1 & lab1_2 scripts.
    [posterior_x1(i,:), g_x1(i,:)] = lab1_2(trainingSet(i, 1), trainingSet); 
    [posterior_x2(i,:), g_x2(i,:)]= lab1_1(trainingSet(i, 2), trainingSet);
    
    %the discriminant of x1 & x2 are being appended to the 6th and 7th
    %columns of the training set to make for easier comparison.
    trainingSet(i, 6) = g_x1(i);
    trainingSet(i, 7) = g_x2(i);
    
    %the following logic is being used to determine whether or not the
    %correct decision has been made
    if((trainingSet(i,6) > threshold_x1 && trainingSet(i,5) == 1) || (trainingSet(i,6) < threshold_x1 && trainingSet(i,5) == 2))
       error_x1 = error_x1 + 1
    end
    if((trainingSet(i,7) > threshold_x2 && trainingSet(i,5) == 1) || (trainingSet(i,6) < threshold_x2 && trainingSet(i,5) == 2))
      error_x2 = error_x2 + 1;
    end
end

%Here the percent error is being calculated
error_x1 = 1 - error_x1 / length(trainingSet)
error_x2 = 1 - error_x2 / length(trainingSet)

