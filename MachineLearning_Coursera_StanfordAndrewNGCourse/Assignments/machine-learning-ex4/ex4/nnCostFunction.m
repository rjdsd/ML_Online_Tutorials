function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add ones to the X data matrix
%  Part 1
X = [ones(m, 1) X];
a_1 = X;
z_2 = a_1*Theta1';
a_2 = sigmoid(z_2);
a_2 = [ones(size(a_2), 1) a_2];
z_3 = a_2*Theta2';
a_3 = sigmoid(z_3);
H = a_3;
H_size = size(H);
I = eye(num_labels);
%size(I)
Y = zeros(m,num_labels);
%size(Y)
for i = 1:m
	Y(i, :) = I(y(i),:);		
end	
J = 1/m * sum(sum(-Y.*log(H) - (1 -Y).* log(1-H))); % without regularization
% element wise square and add each element of Theta1 and Theta2
R = sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2));
R = lambda/(2*m)*R;
J = J + R;

% 	Part 2

for t = 1:m
	a1_t = X(t,:);
	z2_t = a1_t * Theta1';
	a2_t = sigmoid(z2_t);
	a2_t = [1 a2_t];
	z3_t = a2_t *Theta2';
	a3_t = sigmoid(z3_t);
	H = a3_t;
	%H_size = size(H);
	delta_3 = a3_t - Y(t,:);
	%delta_3_size = size(delta_3)
	delta_2 = (delta_3*Theta2);
	% removing the term for bias
	delta_2 = delta_2(2:end).*sigmoidGradient(z2_t);
	Theta2_grad = Theta2_grad + delta_3'*a2_t;  
	Theta1_grad = Theta1_grad + delta_2'*a1_t;
end	

Theta2_grad = (1/m) * Theta2_grad;
Theta1_grad = (1/m) * Theta1_grad;



Theta1_grad(:, 2:end) =  Theta1_grad(:, 2:end) + (lambda/m)*Theta1(:, 2:end);
Theta2_grad(:, 2:end) =  Theta2_grad(:, 2:end) + (lambda/m)*Theta2(:, 2:end);










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
