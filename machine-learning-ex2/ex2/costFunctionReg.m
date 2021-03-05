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

% 先计算损失函数
n = length(theta);    % 计算theta的项数，因为正规化的时候要取出第一项
prediction = sigmoid(X * theta);
regularized_term = lambda * (1 / (2 * m)) * sum(theta(2:n) .^ 2);
J = -(1 / m) * (y' * log(prediction) + (1 - y)' * log(1 - prediction)) + regularized_term;

% 计算梯度
total_error = prediction - y;
derivative_regularized_term = (1 / m) * lambda * theta;
derivative_regularized_term(1) = 0;
grad = (1 / m) * (X' * total_error) + derivative_regularized_term;




% =============================================================

end
