%% load dataset
load('other_model_dataset.mat', 'result');
%% load each group
% G = 6;
% t = result.t_rst{G};
% s = result.s_gt{G};
% t(1)=0.1;
% lg_t = log10(t);
% lg_s = log10(s);
% plot(lg_t, lg_s)
%% fit time-s
logis_type = fittype('aph+(omg-aph)./(1+exp(4*sgm*(ta-x)/(omg-aph)))',...
    'independent','x','coefficients',{'aph','omg','sgm','ta'});
lb = [-Inf,-Inf,-Inf,2;-Inf,-Inf,-Inf,2;-Inf,-Inf,-Inf,1.8;...
    -Inf,-Inf,-Inf,1.6;-Inf,-Inf,-Inf,1.7;-Inf,-Inf,-Inf,1.9;];
ub = [0,-0.65,0,2.5;0,-3.7,0,2.5;0,-0.48,0,2.5;...
    0,-4,0,2.1;0,-1.3,0,2.7;0,-0.7,0,3];
for i = 1:6
    t = result.t_rst{i};
    s = result.s_gt{i};
    t(1)=0.1;
    lg_t = log10(t);
    lg_s = log10(s);
    [res, gof] = fit(lg_t',lg_s',logis_type,...
        'Lower',lb(i,:),'Upper',ub(i,:));
    lss{i} = res;
    gofs{i} = gof;
end
%% decide params
for i = 1:6
    alpha(i) = lss{i}.aph;
    omega(i) = lss{i}.omg;
    sigma(i) = lss{i}.sgm;
    tau(i) = lss{i}.ta;
end
aph = mean(alpha)
omg = mean(omega)
sgm = mean(sigma)
ta = [70 90 80 110 70 70];
rsm = [0.1 0.1 0.2 0.2 0.2 0.2];
sp = [0.75 0.75 0.75 0.75 0.45 0.45];
vd = [2 2 2 2 1 2];
X = [ones(6,1) ta' sp' rsm' vd'];
sol = rref([X'*X X'*tau']);
sol = sol(:,end)
pred_tau = X*sol;
% plot(tau,pred_tau,'o')
f_logistic = @(x,ta)aph+(omg-aph)./(1+exp(4.*sgm.*(ta-x)./(omg-aph)));
%% fit acc
figure(1);
pred_tau = X*sol;
for i = 1:6
    t = result.t_rst{i}';
    s = result.s_gt{i}';
    lg_s = log10(s);
    tt = log10(1:t(end));
    tt = [-1,tt]';
    pred_lgs = f_logistic(tt, pred_tau(i));
    tt = round(10.^tt);
    tt(1) = 0;
    r2_lg = rsquare(lg_s, pred_lgs(t+1));
    fit_lg_acc(i) = r2_lg;
    subplot(2,3,i);
    plot(tt,pred_lgs);
    hold on;
    plot(t, lg_s,'o');
    title(num2str(r2_lg,4));
    pred_s = 10.^pred_lgs;
    r2_nm = rsquare(s,pred_s(t+1));
    fit_nm_acc(i) = r2_nm;

end
fprintf('r2_fit_nm=%.4f \nr2_fit_lg=%.4f \n',...
    mean(fit_nm_acc),mean(fit_lg_acc));
%% test acc
figure(2);
ta_test = [110;90;70]; rsm_test = [0.1;0.2;0.2];
sp_test =[0.75;0.75;1]; vd_test = [2;2;2];
X_test = [ones(3,1),ta_test,sp_test,rsm_test,vd_test];
pred_tau_test = X_test*sol;
for i = 1:3
    t = result.t_test{i}';
    s = result.s_test{i}';
    lg_s = log10(s);
    tt = log10(1:t(end));
    tt = [-1,tt]';
    pred_lgs = f_logistic(tt, pred_tau_test(i));
    tt = round(10.^tt); 
    tt(1) = 0;
    r2_lg = rsquare(lg_s, pred_lgs(t+1));
    test_lg_acc(i) = r2_lg;
    pred_s = 10.^pred_lgs;
    r2_nm = rsquare(s,pred_s(t+1));
    test_nm_acc(i) = r2_nm;
    subplot(1,3,i);
    plot(tt,pred_s);
    hold on;
    plot(t, s,'o');
    title(num2str(r2_nm,4));
end
fprintf('r2_test_nm=%.4f \nr2_test_lg=%.4f \n',...
    mean(test_nm_acc),mean(test_lg_acc));