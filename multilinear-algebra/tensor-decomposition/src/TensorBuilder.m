classdef TensorBuilder
    %UNTITLED5 Summary of this class goes here
    %   Detailed explanation goes here

    methods (Static)

        function X = from_matrix(A, mode, dims)
            % Rearrange the dimensions so that the specified mode becomes the first mode
            perm = [mode, 1:mode-1, mode+1:numel(dims)];
            
            % Reshape the matrix back into the permuted dimensions
            reshaped_dims = [dims(mode), dims(perm(2:end))];
            reshaped_array = reshape(A, reshaped_dims);
            
            % Inverse permute the dimensions to get back to the original order
            reshaped_array = ipermute(reshaped_array, perm);
            
            % Flatten the array
            flattened_array = reshaped_array(:);
            
            % Create an instance of the Tensor class and return it
            X = Tensor(flattened_array', dims);
        end 

        function X = from_cp(lambda, varargin)
        % Rebuild the tensor from the CP decomposition
        % lambda: the vector of weights for each component
        % varargin: the factor matrices
    
        % Initialize the reconstructed tensor to zeros with appropriate dimensions
        dims = cellfun(@(x) size(x, 1), varargin); % Get the dimensions of the tensor
        entries = zeros(dims);  % Initialize tensor with correct dimensions
    
        % Iterate through the rank
        for r_idx = 1:length(lambda)
            % Initialize a cell to hold the r_idx column of each matrix
            columns = cellfun(@(M) M(:, r_idx), varargin, 'UniformOutput', false);
    
            % Concatenate these columns to form the outer product
            entries = entries + lambda(r_idx) * outer_product(columns{:});
        end
    
        X = Tensor(entries(:)', dims);
        end

        function X = from_tucker(G, varargin)
            X = G;
        
            for i = 1:length(varargin)
                 X = X.times(varargin{i}, i);
            end
        end

    end
end