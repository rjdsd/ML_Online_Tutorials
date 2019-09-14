function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
row_count = size(z)(1);
col_count = size(z)(2);
for row = 1:row_count
	for col = 1:col_count
		g(row,col) = 1/(1+exp(-z(row,col)))	;
	end
end

% =============================================================

end
