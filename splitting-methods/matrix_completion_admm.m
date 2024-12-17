function [X, Y, Z] = matrix_completion_admm(B, Omega, mu, lambda, max_iter, tol)
% Solves the low-rank matrix completion problem using ADMM
% Following the theoretical setup:
% min_{X,Y} mu||Y||_N + (1/2)||P_Omega(X) - P_Omega(B)||_F^2
% subject to X >= 0, X - Y = 0
%
% Input:
%   B: Original matrix with known entries
%   Omega: Binary matrix indicating known entries (1 for known, 0 for unknown)
%   mu: Regularization parameter
%   lambda: ADMM penalty parameter
%   max_iter: Maximum iterations
%   tol: Convergence tolerance

% Get dimensions
[m, n] = size(B);

% Initialize variables
X = zeros(m, n);
Y = zeros(m, n);
Z = zeros(m, n);

% Define projection operators
P_Omega = @(X) X .* Omega;      % Projection onto known entries
P_Omegac = @(X) X .* (1-Omega); % Projection onto unknown entries
P_plus = @(X) max(X, 0);        % Projection onto nonnegative orthant
P_Omega_B = P_Omega(B);
Y_prev = Y; 

% Main ADMM loop
for iter = 1:max_iter
    % X-update :
    % X_{k+1} = P_+((1/(1+λ))P_Ω(B + λY_k - Z_k)) + P_+(P_Ωc(Y_k - (1/λ)Z_k))
    
    % First term of xk: (1/(1+λ))P_Ω(B + λY_k - Z_k)
    term1 = (1/(1 + lambda)) * P_Omega(B + lambda*Y - Z);
    
    % Second term of xk : P_Ωc(Y_k - (1/λ)Z_k)
    term2 = P_Omegac(Y - (1/lambda)*Z);
    
    % Combine terms and project onto nonnegative orthant
    X = P_plus(term1) + P_plus(term2);
    
    % Y-update :
    
    % Y_{k+1} = prox_{(μ/λ)||·||_N}(X_{k+1} + (1/λ)Z_k)
    [U, S, V] = svd(X + (1/lambda)*Z, 'econ');
    % Apply soft thresholding to singular values
    s = diag(S);
    s_thresh = max(s - mu/lambda, 0);
    Y = U * diag(s_thresh) * V';
    
    % Z-update (dual variable update):
    % Z_{k+1} = Z_k + λ(X_{k+1} - Y_{k+1})
    Z_prev = Z;
    Z = Z + lambda*(X - Y);
    
    % Check convergence
    primal_res = norm(X - Y, 'fro');
    dual_res = lambda * norm(Y - Y_prev, 'fro');
    
    if primal_res < tol && dual_res < tol

        fprintf('Converged at iteration %d\n', iter);
        fprintf('Primal residual: %e\n', primal_res);
        fprintf('Dual residual: %e\n', dual_res);
        break;
    end
    % Store Y for next iteration's dual residual
    Y_prev = Y;
end
end