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
h_theta_x = sigmoid(X * theta);
J = (1/m)  * sum(-y.*log(h_theta_x) - (1 -y).* log(1- h_theta_x));
grad_row_count = size(theta)(:,1);
for row = 1:grad_row_count
	grad(row) = 1/m * sum((h_theta_x - y) .* X(:,row));
end
%grad(1) = 1/m * sum((h_theta_x - y) .* X(:,1));
%grad(2) = 1/m * sum((h_theta_x - y) .* X(:,2));
%grad(3) = 1/m * sum((h_theta_x - y) .* X(:,3));




% =============================================================

end
