testcases = cell(1, 8);

testcases{1} = Tensor([1:18], [3 3 2]);
[lambda, factors] = Jennrich.decompose(testcases{1});