classdef Jennrich
    methods (Static)
        function [exit_code, lambda, factors] = decompose(T)
            Mx = Jennrich.composed_matrix(T);
            My = Jennrich.composed_matrix(T);
            lambda = 0;
            factors = 0;

            [A, same] = Jennrich.nonzero_eigvectors(Mx*pinv(My));
            
            if same == 1
                exit_code = 1;
                fprintf("A is not full column-rank, exit...\n");
            else
                B = Jennrich.nonzero_eigvectors(Mx'*pinv(My'));
                C = linsolve(katri_rao(B, A), T.to_matrix(3)')';
                
                computed_rank = size(A, 2);
                fprintf("Rank determined by Jennrich's: %d\n", computed_rank);
    
                lambda = zeros(1, computed_rank);
    
                for i = 1:computed_rank
                    normA = norm(A(:, i));
                    A(:, i) = A(:, i) / normA;
    
                    normB = norm(B(:, i));
                    B(:, i) = B(:, i) / normB;
    
                    normC = norm(C(:, i));
                    C(:, i) = C(:, i) / normC;
    
                    lambda(i) = normA*normB*normC;
                end
    
                factors = {A, B, C};
                exit_code = 0;
            end

           
        end

        function M = composed_matrix(T)

            if length(T.dims) ~= 3
                fprintf("Jennrich's requires order-3 tensor");
            end

            v = Jennrich.random_unit_vector(T.dims(3));

            M = zeros(T.dims(1), T.dims(2));

            for i = 1:T.dims(3)
                M = M + T.frontal_slice(i)*v(i);
            end

        end

        function v = random_unit_vector(dim)
            v = abs(randn(dim, 1)); 
            v = v / norm(v);
        end

        function [P, same] = nonzero_eigvectors(A, tol)
            arguments
                A;
                tol = 1e-5;
            end
            
            [P, D] = eig(A);

            [v,ind] = sort(real(diag(D)));
            P_sorted = P(:,ind);

            cur = 0;
            same = 0;

            for i = 1:length(v)
                if cur > 0 && abs(cur-v(ind(i))) < tol
                    fprintf("There are two equal eigenvalues!\n");
                    same = 1;
                    break;
                end

                cur = v(ind(i));
            end
            
            if same == 0
                for i = 1:length(v)
                    if v(i) > tol
                        P = real(P_sorted(:, i:end));
                        break;
                    end
                end
            end

        end
    end
end