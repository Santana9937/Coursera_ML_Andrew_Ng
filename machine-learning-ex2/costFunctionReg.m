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


% Sigmoid function argument. size(X)=(m, n+1) and size(theta) = (n+1,1)
% Thus, size(z)=(m,1)
z = X*theta ; 
% Hypothesis function is size (m,1)
htheta = sigmoid(z) ; 
% Cost function for unregularized part is the summation of two (m,1) 
% vectors over all m observation. Can accomplish this with a dot product.
% Cost function for the regularized part is the summation of two (n,1) 
% vectors over n features. Intercept feature is note regularized. 
% Can accomplish this with a dot product.
J = (1.0/m).*( -transpose(y)*log(htheta) - transpose((1.0-y))*log(1.0-htheta) ) + ...
    (lambda/(2.0*m))*( transpose(theta(2:end))*theta(2:end) ) ;

% Error between hypothesis and observations size = (m,1)
errors = htheta - y ;
% Size of gradient is (n+1,1). size(X)=(m,n+1). size(errors)=(m,1)
% Unregularized part is the same as before. For the regularized part,
% make sure it is 0 for theta_0 and theta otherwise. The vector 
% [0; theta(2:end)] accomplishes this.
grad = (1.0/m).*( transpose(X)*errors ) + (lambda/m)*[0; theta(2:end)] ; 



% =============================================================

end
