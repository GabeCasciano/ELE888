%% Lab 2
%%
% Gabriel Casciano, 500744076
%%
% Aleksandar Nikolovski, 500572171
%%
% Khayyam Butt, 500501040
%%
% The subject of study for this lab is Linear Discrimnant functions.
%%
% The training set being used is the iris training set.<

%% Question 1
% Construct a new data set from *A* and *B* by setting aside 30% of the
% sample from those original sets (this will be used for _training
% purposes_). Construct a second set from the remaining 70% of data (this
% will be used for _testing_). Use the 30% set (training set) to compute
% the weight vector $\eta(k) = \eta(\bullet) = 0.01, \Theta = 0,$ and an
% initial values of $a(0) = [001]$. Limit your maximum iterations to 300
% (i.e even if you can not achieve $\theta=0$).

    trainset1 = zeros(30,3); %   30% Set
    trainset2 = zeros(70,3);  %   70% Set

    % Creates a set consistining of 15 samples from Set A and 15 from Set B,
    % resulting in 30 samples for the 30% Set
    for i = 1:15
        trainset1(i,:) = [dataset(i,2), dataset(i,3), dataset(i,5)];
    end
    j = 16;
    for i = 1:15
        trainset1(j,:) = [dataset(i+50,2), dataset(i+50,3), dataset(i+50,5)];
        j = j + 1;
    end
    % Creates a set of 35 samples from Set A and 35 Samples from Set B
    % resulting in a 70 sample set for the 30% Set
    j = 1;
    for i = 16:50
        trainset2(j,:) = [dataset(i,2), dataset(i,3), dataset(i,5)];
        j = j + 1;
    end
    j = 36;
    for i = 16:50
        trainset2(j,:) = [dataset(i+50,2), dataset(i+50,3), dataset(i+50,5)];
        j = j + 1;
    end

    %% Computation
    eta = 0;    theta = 0;  a = 0;  y = 0;  Jp = 0; check = 0;

    eta = 0.01;
    theta = 0;
    a = [0 0 1]';
    y = zeros(15,3);
    Jp = [0 0 0]';
    check = zeros(3,1);
    % Augmentation

    for i = 1:30
        y(i,:) = [1, trainset1(i,1),trainset1(i,2)];
    end
    y = y';
    % Normalization
    for i = 16:30
        y(:,i) = (-1).*y(:,i);
    end

    k = 0;
    while(1)
        ay = (a')*y;
        for i = 1:30
            m = 0;
            if (ay(:,i) <= 0)
                Jp = Jp + (-1)*y(:,i);
            elseif(ay(:,i) > 0)
                m = m+1;
            end
            if m == 30
                break;
            end
        end
        a = a - eta*Jp;
        check = abs(eta*Jp);

        if(check(1)<theta && check(2)<theta && check(3)<theta)
            break;
        elseif(k > 300)
            break;
        end
        k = k + 1;
    end
    
    
%% Questions 2
% Use the 70% set (test samples) and the weight vector computer in the
% previous step to calculate the classification accuracy of the classifier.

%% Question 3
% Repeat the above steps by changing the data split to 70% (training) and
% 30% (testing).
    %% Computation
    eta = 0;    theta = 0;  a = 0;  y = 0;  Jp = 0; check = 0;

    eta = 0.01;
    theta = 0;
    a = [0 0 1]';
    y = zeros(70,3);
    Jp = [0 0 0]';
    check = zeros(3,1);
    % Augmentation

    for i = 1:70
        y(i,:) = [1, trainset2(i,1),trainset2(i,2)];
    end
    y = y';
    % Normalization
    for i = 36:70
        y(:,i) = (-1).*y(:,i);
    end

    k = 0;
    while(1)
        ay = (a')*y;
        for i = 1:70
            m = 0;
            if (ay(:,i) <= 0)
                Jp = Jp + (-1)*y(:,i);
            elseif(ay(:,i) > 0)
                m = m + 1;
            end
            if m == 30
                break;
            end
        end
        a = a - eta*Jp;
        check = abs(eta*Jp);

        if(check(1)<theta && check(2)<theta && check(3)<theta)
            break;
        elseif(k > 300)
            break;
        end
        k = k + 1;
    end
    
%% Question 4
% Repeat all the above for test and training sets constructed from the
% combination of data set *B* and *C* (i.e _Iris Versicolor_ vs, _Iris
% Virginica_)

%% Question 5
% Compute the weight vextors and stuct the gradient descent algorithms for
% two different values of $\eta(k)$, and the initial values of *a*.

%% Question 6
% 
