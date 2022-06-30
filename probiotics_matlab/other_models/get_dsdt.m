load('other_model_dataset.mat');
df = @(f,x)(f(x+0.01)-f(x-0.01))/0.002;
n_tv = [1 2 4 6 7 8];
for i = 1:6
    s = result.s_gt{i};
    t = result.t_rst{i};
    pch = fit(t',s','pchip');
    dsdt = df(pch, t);
    data = [s',dsdt];
    xlswrite('dsdt.xlsx',data,['G',num2str(n_tv(i))]);
end