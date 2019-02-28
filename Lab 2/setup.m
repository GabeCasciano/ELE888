clear ;
load irisdata.mat;

%Initiating the whole data set with numberical labels
dataset = irisdata_features;
for i = 1 : size(irisdata_features)
    if strcmp('"Iris-setosa"', irisdata_labels(i)) 
        dataset(i, 5) = 1;
    elseif strcmp('"Iris-versicolor"', irisdata_labels(i)) 
        dataset(i, 5) = 2;
    elseif strcmp('"Iris-virginica"', irisdata_labels(i)) 
        dataset(i, 5) = 3;
    end
end

%Initializing datasets A, B, C with features x2 & x3, as well as the
%associated numeric labels
datasetA = zeros(50,3);
datasetB = zeros(50,3);
datasetC = zeros(50,3);
for i = 1 : 50
   datasetA(i,:) = [dataset(i,2), dataset(i,3), dataset(i,5)];
   datasetB(i,:) = [dataset(i+50,2), dataset(i+50,3), dataset(i+50,5)];
   datasetC(i,:) = [dataset(i+100,2), dataset(i+100,3), dataset(i+100,5)];
end

%Splitting the datasets into their required partitions
datasetAB = [datasetA; datasetB];
ABtraining = [datasetAB(1:15,1:3); datasetAB(86:100,1:3)];
ABtesting = datasetAB(16:85, 1:3);
datasetBC = [datasetB; datasetC];
BCtraining = [datasetBC(1:15,1:3); datasetBC(86:100,1:3)];
BCtesting = datasetBC(16:85, 1:3);

%Other Variables
err = 0;