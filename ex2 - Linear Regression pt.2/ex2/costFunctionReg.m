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
J = 0;
for i = 1:m
  h = sigmoid(theta'*X(i,:)')
  J += ((-y(i)*log(h)) - (1-y(i))*log(1-h));
endfor

alpha = 0
for j = 2:length(theta)
  alpha += theta(j)**2
endfor

alpha = (lambda/(2*m))*alpha
J = J/(m) + alpha;

grad = zeros(size(theta));
for i = 1:m
  h = sigmoid(theta'*X(i,:)')
  grad += ((h - y(i))*X(i, :)');
endfor
grad = grad/m

for j = 2:length(theta)
  grad(j) += (lambda/m)*theta(j);
endfor




% =============================================================

end
