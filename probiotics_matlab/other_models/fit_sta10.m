ress = {res1,res2,res4,res6,res7,res8};
for i = 1:6
    alpha(i) = ress{i}.alpha;
    beta(i) = ress{i}.beta;
end
Ta=[70,90,70,110,70,70]+273.15;
cr=[0.1,0.1,0.2,0.2,0.2,0.2];
v = [ones(1,4)*0.75,0.45,0.45];
lg_alpha = log10(alpha);
inp = [Ta',cr',v'];