function O = outer_product(varargin)
    % Get the number of input vectors
    n = nargin;
    
    % Check if there are any vectors
    if n == 0
        error('At least one vector must be provided.');
    end
    
    % Get the sizes of the input vectors
    sizes = cellfun(@(vec) size(vec, 1), varargin);

    reversed_factors = flip(varargin);

    % Unpack varargin and pass to the kronecker function
    O = reshape(kronecker(reversed_factors{:}), sizes);
end
