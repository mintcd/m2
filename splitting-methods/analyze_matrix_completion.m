function analyze_matrix_completion()
    % Fixed dimensions
    m = 50;  % rows
    n = 40;  % columns
    r = 3;   % rank
    
    % Create exact rank-r matrix without max operation
    U = randn(m, r);
    V = randn(n, r);
    B_true = U * V';  % Remove max operation for exact rank
    
    % Test different parameters
    mus = [0.1, 0.5, 1.0, 3.0, 5.0];  % Different regularization parameters
    lambdas = [0.1, 0.5, 1.0, 2.0];    % Different ADMM parameters
    sample_rates = [0.5]  ; 
    % Store results
    results = zeros(length(mus), length(lambdas), length(sample_rates));
    
    % Run experiments
    for i = 1:length(mus)
        for j = 1:length(lambdas)
            for k = 1:length(sample_rates)
                % Create observation mask
                Omega = rand(m, n) < sample_rates(k);
                B = B_true .* Omega;
                
                % Run ADMM
                [X, ~, ~] = matrix_completion_admm(B, Omega, mus(i), lambdas(j), 10000, 1e-6);
                
                % Calculate relative error
                rel_error = norm(X - B_true, 'fro') / norm(B_true, 'fro');
                results(i,j,k) = rel_error;
                
                % Print results
                fprintf('μ = %.1f, λ = %.1f, sample_rate = %.1f: Relative Error = %e\n', ...
                    mus(i), lambdas(j), sample_rates(k), rel_error);
            end
        end
    end
    
    % Find best parameters
    [min_error, idx] = min(results(:));
    [i_best, j_best, k_best] = ind2sub(size(results), idx);
    
    fprintf('\nBest parameters found:\n');
    fprintf('μ = %.1f\n', mus(i_best));
    fprintf('λ = %.1f\n', lambdas(j_best));
    fprintf('Sample rate = %.1f\n', sample_rates(k_best));
    fprintf('Best relative error: %e\n', min_error);
    
    % Plot results for best sampling rate
    figure;
    [X, Y] = meshgrid(lambdas, mus);
    surf(X, Y, results(:,:,k_best));
    xlabel('λ');
    ylabel('μ');
    zlabel('Relative Error');
    title(sprintf('Error Surface (Sample Rate = %.1f)', sample_rates(k_best)));
    colorbar;
end