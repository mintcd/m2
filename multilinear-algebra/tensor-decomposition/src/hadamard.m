function X = hadamard(varargin)
    % HADAMARD Computes the Hadamard product of multiple matrices.
    % 
    % Syntax:
    %   X = hadamard(A, B, ...)
    %
    % Inputs:
    %   - A, B, ... : Matrices of the same size.
    %
    % Outputs:
    %   - X : Matrix of the same size as the inputs, containing the 
    %         element-wise product of all input matrices.

    dims = size(varargin{1});
    X = varargin{1};
    for i=2:nargin
        if size(varargin{i}) ~= dims
            error("All matrices must have the same size")
        end
        X = X .* varargin{i};
    end
end