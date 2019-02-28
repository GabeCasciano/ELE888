function [w, Jp, conv, fin] = gradientDescent(a, theta, y, eta, iter, prim)
%Gradient Descent,  this function will perform gradient descent on
%a given set of data. 

%a     -> inital augmented weight vector
%theta -> threshold
%y     -> augmented feature vector
%eta   -> learning rate
%iter  -> number of iterations
    
    temp = zeros(length(y), 4);
    grad = zeros(iter, 3);
    conv = 0;
    accum = 0;
    i = 1;
    flag = 0;
    %augmentation
    for(k = 1 : length(y))
        temp(k,:) = [1,y(k,:)];
    end
    y = temp;
        
    while(i < iter  && flag ~=1)
        
        accum = 0;
        for j = 1 : length(y)
           class = y(j, 4);
           feature = y(j,1:3);
           if(class ~= prim)
               feature = feature .* -1;
           end
           if(classifier2(a, feature) ~= 1)
               grad(i,:) = grad(i,:) + feature;
               %gradient = gradient + missClassed(accum, 1:3);
               accum = accum + 1;
           end 
        end
        a = a + eta .* grad(i,:);
        if(norm(eta.* grad(i,:)) <= theta)
           flag = 1;
           conv = i-2;
        else
           conv = iter;
       end
       i = i + 1;
    end
    w = a;
    Jp = grad(1:conv,:);
    fin = grad(conv-1,:);
    %plot(grad)
    %plot3(grad(:,1),grad(:,2),grad(:,3))
end


