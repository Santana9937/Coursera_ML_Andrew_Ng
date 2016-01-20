function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

% Sigmoid function argument. size(X)=(m, n+1) and size(theta) = (n+1,1)
% Thus, size(z)=(m,1)
z = X*theta ; 
% Hypothesis function is size (m,1)
htheta = sigmoid(z) ; 
% Cost function is the summation of two (m,1) vectors over all m
% observation. Can accomplish this with a dot product.
J = (1.0/m).*( -transpose(y)*log(htheta) - transpose((1.0-y))*log(1.0-htheta) ) ;

% Error between hypothesis and observations size = (m,1)
errors = htheta - y ;
% Size of gradient is (n+1,1). size(X)=(m,n+1). size(errors)=(m,1)
grad = (1.0/m).*( transpose(X)*errors ) ; 
 
% =============================================================

end
