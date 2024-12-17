testcases = cell(1, 2);
tol = 1e-5;

testcases{1} = Tensor([1 2 2 4 4 8 8 16], [2 2 2]);
testcases{2} = Tensor(1:18, [3 3 2]);
testcases{3} = Tensor(1:32, [4 4 2]);
A
for i = 1:length(testcases)
    fprintf("-------------------------------------- \n");
    fprintf("Testcase %d \n", i);
    testcases{i}.cp();

    fprintf("Runtime 1...\n")
    [exit_code, lambda, factors] = Jennrich.decompose(testcases{i});
    if exit_code == 1
        fprintf("Jennrich's is unable to solve this case!\n")
    else
        reconstructed = TensorBuilder.from_cp(lambda, factors);
        difference = testcases{i} - reconstructed;
    
        if difference.norm_squared() > tol
            % Run one more time
            fprintf("Runtime 2...\n")
            [lambda, factors] = Jennrich.decompose(testcases{i});
            reconstructed = TensorBuilder.from_cp(lambda, factors);
            difference = testcases{i} - reconstructed;
    
            if difference.norm_squared() > tol
                fprintf("Jennrich's is unable to solve this case!\n")
            else
                fprintf("Difference: %f\n", difference.norm_squared());
            end
        else
            fprintf("Difference: %f\n", difference.norm_squared());
        end
    end
    
end


