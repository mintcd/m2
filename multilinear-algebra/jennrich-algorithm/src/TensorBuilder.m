classdef TensorBuilder

    methods (Static)

        function X = from_cp(lambda, factors)
        % Rebuild the tensor from the CP decomposition
        % lambda: the vector of weights for each component
        % varargin: the factor matrices
        
        dims = cellfun(@(x) size(x, 1), factors); % Get the dimensions of the tensor
        entries = zeros(dims);  % Initialize tensor with correct dimensions
    
        % Iterate through the rank
        for r_idx = 1:length(lambda)
            % Initialize a cell to hold the r_idx column of each matrix
            columns = cellfun(@(M) M(:, r_idx), factors, 'UniformOutput', false);
    
            % Concatenate these columns to form the outer product
            entries = entries + lambda(r_idx) * outer_product(columns{:});
        end
        
        X = Tensor(entries(:)', dims);
        end

    end
end