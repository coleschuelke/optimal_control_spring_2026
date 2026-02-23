clear variables;
close all;
clc;

syms x1 x2 lam
vars = [x1, x2, lam];

h = 1/2 - x1*x2;

gf1 = x1;
gf2 = 1/4*x2;
gh1 = -x2;
gh2 = -x1;

gf = [gf1, gf2];
gh = [gh1, gh2];

Hf = [1, 0; 0, 1/4];
Hh = [0, -1; -1, 0];

gl = gf + lam*gh;

candidates = solve([gl == [0, 0], h == 0], vars);

Hl = Hf + lam*Hh;

Hlx0 = zeros(2, 2, length(candidates.x1));
for i=1:length(candidates.x1)
    soli = [candidates.x1(i), candidates.x2(i), candidates.lam(i)];
    Hlx0(:, :, i) = subs(Hl, vars, soli);
    disp(soli)
    disp(Hlx0(:, :, i))
    eigi = eigs(Hlx0(:, :, i));
    disp(eigi)
    
    temp = subs(gh, vars, soli);
    ghnum = double(temp);
    [Q, R] = qr(ghnum.');
    E = Q(:, 2:end);

    HlTSi = E.'*Hlx0(:, :, i)*E
    eig2i = eig(HlTSi)
end

