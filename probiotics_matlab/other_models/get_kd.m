df = @(f,x)(f(x+0.01)-f(x-0.01))/0.002;
f_kd = @(fit,f_t,f_s,ind)-df(fit,f_t(ind))'./f_s(ind);
% flatten = @(x)reshape(x,1,numel(x));
n_tv = [1 2 4 6 7 8];
for i = 1:6
    s = result.s_gt{i};
    t = result.t_rst{i};
    pch = fit(t',s','pchip');
    kd = f_kd(pch,t,s,1:length(s));
    data = [t;kd];
    % 因为要取对数，所以把非正数kd舍弃
    data(:, data(2,:)<=0) = [];
    xlswrite('kd.xlsx',data',['G',num2str(n_tv(i))]);
end