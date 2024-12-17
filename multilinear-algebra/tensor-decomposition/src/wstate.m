u = zeros(2,1);
v = zeros(2,1);

u(1) = rand();
u(2) = sqrt(1-u(1)^2);

v(1) = rand();
v(2) = sqrt(1-v(1)^2);


tensor = kronecker(u, u, v) + kronecker(u, v, u) + kronecker(v, u, u);
tensor = Tensor(tensor', [2 2 2]);
tensor.cp()