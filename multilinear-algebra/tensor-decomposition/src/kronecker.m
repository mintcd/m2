function C = kronecker(varargin)
    % Check the number of input arguments
    numMatrices = nargin;
    if numMatrices < 2
        error('At least two matrices are required for the Kronecker product.');
    end

    % Initialize the result matrix C as the first matrix
    C = varargin{1};

    % Compute the Kronecker product iteratively for each additional matrix
    for k = 2:numMatrices
        C = kroneckerTwo(C, varargin{k});
    end
end

function C = kroneckerTwo(A, B)
    % Get the dimensions of matrices A and B
    [m, n] = size(A);
    [p, q] = size(B);
    
    % Initialize the result matrix C
    C = zeros(m * p, n * q);
    
    % Compute the Kronecker product
    for i = 1:m
        for j = 1:n
            % Each element of A is multiplied by the entire matrix B
            C((i-1)*p+1:i*p, (j-1)*q+1:j*q) = A(i, j) * B;
        end
    end
end
