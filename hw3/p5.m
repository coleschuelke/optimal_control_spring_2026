clear variables;
close all;
clc;

syms x1 x2 x3 lam
vars = [x1, x2, x3, lam];

h = x1^2 + x2^2 + x3^2 - 1;

gf1 = 2;
gf2 = 4*x2 + 2*x3;
gf3 = 2*x2+8*x3;
gh1 = 2*x1;
gh2 = 2*x2;
gh3 = 2*x3;

gf = [gf1, gf2, gf3];
gh = [gh1, gh2, gh3];

Hf = [0, 0, 0;
    0, 4, 2;
    0, 2, 8];
Hh = [2, 0, 0;
    0, 2, 0;
    0, 0, 2];

gl = gf + lam*gh;

candidates = solve([gl == [0, 0, 0], h == 0], vars);

Hl = Hf + lam*Hh;

Hlx0 = zeros(3, 3, length(candidates.x1));
for i=1:length(candidates.x1)
    soli = [candidates.x1(i), candidates.x2(i), candidates.x3(i), candidates.lam(i)];
    Hlx0(:, :, i) = subs(Hl, vars, soli);
    disp(soli)
    disp(Hlx0(:, :, i))
    eigi = eigs(Hlx0(:, :, i));
    disp(eigi)
    
    ghnum = double(subs(gh, vars, soli));
    [Q, R] = qr(ghnum.');
    E = Q(:, 2:end);

    HlTSi = E.'*Hlx0(:, :, i)*E
    eig2i = eig(HlTSi)
end