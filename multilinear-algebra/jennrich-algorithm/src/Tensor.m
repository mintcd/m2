classdef Tensor

    properties
        entries
        dims
        nmodes
    end

    methods

    function self = Tensor(entries, dims)
        nEntries = prod(dims);
        if length(entries) ~= nEntries
            error("Number of entries is not compatible with dimensions.")
        end

        self.entries = entries;
        self.dims = dims;
        self.nmodes = length(dims);
    end
    
    function disp(self)
        fprintf('Tensor of dimensions: [%s]\n', num2str(self.dims));
        fprintf('Entries:\n');
        disp(reshape(self.entries, self.dims));
    end

    % Overload common operators
    function Z = minus(A, B)

        if A.dims ~= B.dims
            error("Two tensors must have the same dimensions")
        end
        Z = Tensor(A.entries - B.entries, A.dims);
    end

    function ns = norm_squared(self)
        ns = sum(self.entries.^2);
    end

    function A = frontal_slice(self, n)
        tensor = reshape(self.entries, self.dims);
        A = tensor(:, :, n);
    end

    function A = to_matrix(self, n)
        % Matricizes the tensor in mode n
        n_dim = self.dims(n);
        other_dims = [self.dims(1:n-1), self.dims(n+1:end)];
        other_dim_prod = prod(other_dims);

        permutedDims = [n, 1:n-1, n+1:self.nmodes];

        permutedEntries = permute(reshape(self.entries, self.dims), permutedDims);
        A = reshape(permutedEntries, n_dim, other_dim_prod);
    end

    function Y = times(self, A, n)
        % Multiplies the tensor with matrix A in mode n
    
        if size(A, 2) ~= self.dims(n)
            error("Matrix has %d columns, while mode-%d is of size %d.", ...
                size(A,2), n, self.dims(n));
        end

        newDims = self.dims;

        for i = 1:self.nmodes
            if i == n
                newDims(i) = size(A, 1);
            end
        end
    
        Yn = A * self.to_matrix(n);
        Y = TensorBuilder.from_matrix(Yn, n, newDims);
    end

    function ranks = multilinear_ranks(self)
        ranks = zeros(1, self.nmodes);

        for n = 1:self.nmodes
            ranks(n) = rank(self.to_matrix(n));
        end
    end

    function [lambda, factors, errors] = cp_asl(self, r, epsilon, max_iter)
        % CP_ASL Computes a rank-r CP approximation of a tensor X.
        % Output:
        % - lambda: normalizing vector
        % - factors: factor matrices
        % - converged (0 or 1): specifies if the procedure converges before reaching max_iter
        
        arguments
            self;                % The tensor
            r;                % Rank of the decomposition
            epsilon = 1e-10;   % Convergence tolerance
            max_iter = 5000;  % Maximum number of iterations
        end
    
        % Initialize lambda and factor matrices
        lambda = ones(r, 1);
        factors = cell(1, self.nmodes); % Cell array to store factor matrices
        errors = zeros(1, max_iter);

        % Randomly initialize the factor matrices
        for n = 1:self.nmodes
            factors{n} = rand(self.dims(n), r);
        end

    
        fprintf('Approximating by rank-%d tensor...\n', r);
    
        % Iterate until convergence or reaching the maximum number of iterations
        for iter = 1:max_iter
            for n = 1:self.nmodes
                % Get all factor, except for current mode
                current_factors = factors([1:n-1, n+1:self.nmodes]);
                reversed_factors = flip(current_factors);
                
                gramians = cellfun(@(mat) mat' * mat, current_factors, 'UniformOutput', false);
                V = hadamard(gramians{:});
    
                factors{n} = self.to_matrix(n)*katri_rao(reversed_factors{:})/V;
    
                % Normalize the factor matrix and adjust lambda
                lambda = sqrt(sum(factors{n}.^2, 1))';
                factors{n} = factors{n} ./ lambda';
            end
    
            % Check for convergence
            reconstructed_tensor = TensorBuilder.from_cp(lambda, factors);
            diff_tensor = self - reconstructed_tensor;
            error = diff_tensor.norm_squared();

            if error < epsilon
                fprintf('Converged after %d iterations.\n', iter);
                break;
            end
            errors(iter) = error;
        end

        if iter == max_iter
            fprintf('Reached maximum %d iterations.\n', max_iter);
        end
    
    
        fprintf('Final error %e.\n', error);
        fprintf("------------------------------------------------------------------\n")
        
    end
    
    function [lambda, factors] = cp(self, epsilon, max_iter)
        % CP Decomposition of a tensor X
        % Returns lambda (a vector of weights) and factors (cell array of factor matrices)
        arguments
            self;                % The tensor
            epsilon = 1e-10;   % Convergence tolerance
            max_iter = 5000;  % Maximum number of iterations
        end
    
        fprintf("CP decomposition...\n");
        fprintf("Tensor of dimensions: ");
        disp(self.dims);
    
        [i, j] = ndgrid(1:length(self.dims), 1:length(self.dims));
        % Create a mask for i not equal to j
        mask = i ~= j;
        % Apply the mask using arrayfun
        mutual_products = arrayfun(@(x, y) self.dims(x) * self.dims(y), i(mask), j(mask));
    
        max_rank = min(mutual_products(:));
    
        fprintf("Max possible rank: %d \n", max_rank);
    
        % Initialize a cell array to store the errors for each rank
        all_errors = cell(max_rank, 1);
    
        % Loop over ranks
        for r = 1:max_rank
            [lambda, factors, errors] = cp_asl(self, r, epsilon, max_iter);
            all_errors{r} = errors;  % Store the errors for this rank
            if errors(length(errors)) < epsilon
                break;
            end
        end
    
    end
    end
   
end
