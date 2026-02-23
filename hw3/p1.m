clear variables;
close all;
clc;

syms x1 x2

gf1 = 2*x1+4*x2+2;
gf2 = 2*x2+4*x1-2;
Hf = [2, 4; 4, 2];
eqns = [gf1 == 0, gf2 == 0];
sol = solve(eqns, [x1, x2]);

Hfx0 = zeros(2, 2, length(sol.x1));
for i=1:length(sol.x1)
    Hfx0(:, :, i) = subs(Hf, [x1, x2], [sol.x1(i), sol.x2(i)]);
    disp(Hfx0(:, :, i))

    eigi = eig(Hfx0(:, :, i))
end
