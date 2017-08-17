function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

J = 0;
for i = 1:m
  h = theta'*X(i,:)';
  J += (h-y(i))**2;
endfor

alpha = 0;
for j = 2:length(theta)
  alpha += theta(j)**2;
endfor

alpha = (lambda/(2*m))*alpha;
J = J/(2*m) + alpha;

grad = zeros(size(theta));
for i = 1:m
  h = theta'*X(i,:)';
  grad += ((h - y(i))*X(i, :)');
endfor
grad = grad/m;

for j = 2:length(theta)
  grad(j) += (lambda/m)*theta(j);
endfor










% =========================================================================

grad = grad(:);

end
