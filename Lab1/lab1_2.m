%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ELE 888/ EE 8209: LAB 1: Bayesian Decision Theory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [posteriors_x, g_x]=lab1_2(x,Training_Data)

% x = individual sample to be tested (to identify its probable class label)
% featureOfInterest = index of relevant feature (column) in Training_Data 
% Train_Data = Matrix containing the training samples and numeric class labels
% posterior_x  = Posterior probabilities
% g_x = value of the discriminant function

%% Prior Probabilities

% Hint: use the commands "find" and "length"
setosaNum = 1;
versicolorNum = 1;

for i = 1 : length(Training_Data)
    if Training_Data(i, 5) == 1
        setosa(setosaNum) = Training_Data(i, 1);
        setosaNum = setosaNum + 1;
    end
    if Training_Data(i, 5) == 2
        versicolor(versicolorNum) = Training_Data(i, 1);
        versicolorNum = versicolorNum + 1;
    end
end       

Pr1 = length(setosa)/length(Training_Data);
Pr2 = length(versicolor)/length(Training_Data);

%% Class-conditional probabilities
m11 = mean(setosa);% mean of the class conditional density p(x/w1)
std11 = std(setosa);% Standard deviation of the class conditional density p(x/w1)

m12 = mean(versicolor);% mean of the class conditional density p(x/w2)
std12= std(versicolor);% Standard deviation of the class conditional density p(x/w2)

cp11= (1/(sqrt(2*pi)*std11))*exp(-.5*((x-m11)/std11).^2);% use the above mean, std and the test feature to calculate p(x/w1)
cp12= (1/(sqrt(2*pi)*std12))*exp(-.5*((x-m12)/std12).^2);% use the above mean, std and the test feature to calculate p(x/w2)

%% Compute the posterior probabilities

pos11= Pr1 * cp11 / (cp11 * Pr1 + cp12 * Pr2);% p(w1/x) for the given test feature value

pos12= Pr2 * cp12 / (cp11 * Pr1 + cp12 * Pr2);% p(w2/x) for the given test feature value

posteriors_x= [pos11 pos12];

%% Discriminant function for min error rate classifier%%%


g_x= pos11 - pos12;% compute the g(x) for min err rate classifier.


