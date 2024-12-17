function X = katri_rao(varargin)
    % KATRI_RAO Computes the Katri-Rao product of multiple matrices.
    % 
    % Syntax:
    %   X = hadamard(A, B, ...)
    %
    % Inputs:
    %   - A, B, ... : Matrices having the same number of columns.
    %
    
    % Get the number of columns from the first matrix
    [~, n] = size(varargin{1});
    
    % Check if all matrices have the same number of columns
    for i = 1:nargin
        [~, q] = size(varargin{i});
        if q ~= n
            error('The number of columns of all matrices must be the same');
        end
    end
    
    % Initialize X with the number of rows equal to the product of the number of rows of varargin matrices
    numRows = 1; % Initialize to calculate total rows
    for i = 1:nargin
        [m, ~] = size(varargin{i});
        numRows = numRows * m;
    end

    X = zeros(numRows, n); % Initialize the result matrix
    
    % Iterate over the remaining matrices
    for i = 1:n
        cols = cellfun(@(mat) mat(:, i), varargin, 'UniformOutput', false);
        X(:, i) = kronecker(cols{:});
    end
end
