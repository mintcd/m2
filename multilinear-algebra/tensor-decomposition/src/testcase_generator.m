testcases = cell(1, 6);

% Analytically solvable tensors
testcases{1} = Tensor([0 0 2 2 2 2 0 0], [2 2 2]);  % rank 2
testcases{2} = Tensor([1 0 0 1 0 -1 1 0], [2 2 2]); % rank 3
testcases{3} = Tensor([-1 0 0 1 0 1 1 0], [2 2 2]); % rank 3
testcases{4} = Tensor([1 3 2 6 2 6 4 12], [2 2 2]); % rank 1

% Random tensor
testcases{5} = Tensor(1:8, [2 2 2]);
testcases{6} = Tensor(1:27, [3 3 3]);


for i = 1:length(testcases)
    fprintf("----------- TESTCASE %d ---------------------\n", i)
    cp_res = testcases{i}.cp();

    fprintf("HOSVD...\n")
    hosvd_res = testcases{i}.hosvd();

    max_ranks = testcases{i}.multilinear_ranks;
    ranks = max_ranks;
    for idx = 1:length(ranks)
        if ranks(idx) > 1
            ranks(idx) = ranks(idx) - 1;
        end
    end

    fprintf("HOOI with ranks...\n")

    hooi_res = testcases{i}.hooi(ranks);
end