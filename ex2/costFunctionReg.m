function [J, grad] = costFunctionReg(theta, X, y, lambda)
    %COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    %   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    %   theta as the parameter for regularized logistic regression and the
    %   gradient of the cost w.r.t. to the parameters.

    % Initialize some useful values
    m = length(y); % number of training examples

    % You need to return the following variables correctly
    J = 0;
    grad = zeros(size(theta));

    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the cost of a particular choice of theta.
    %               You should set J to the cost.
    %               Compute the partial derivatives and set grad to the partial
    %               derivatives of the cost w.r.t. each parameter in theta
    H = sigmoid(X*theta);
    summ = (-y.*log(H)) - ((1 - y).*(log(1 - H)));
    J1 = sum(summ) / m;
    theta_sum = 0;
    for n = 2:size(theta)
        theta_sum = theta_sum + theta(n, 1)^2;
    end
    J2 = (lambda/(2*m)) * theta_sum;
    J = J1 + J2;

    grad = (H-y)'*X/m;
    for n = 2:size(theta)
        grad(n) = grad(n) + (lambda/m).*theta(n, 1);
    end



    % =============================================================

end
