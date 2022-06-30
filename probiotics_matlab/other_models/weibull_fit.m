%% load dataset
clear;
load('other_model_dataset.mat', 'result');
%% load each group
% G = 6;
% t = result.t_rst{G};
% s = result.s_gt{G};
% lg_s = log10(s);
% plot(t, lg_s)
%% fit time-s
weibull_type = fittype('-1/2.303*(x/aph).^beta',...
    'independent','x','coefficients',{'aph','beta'});
lb = [0.7,0.01; 0.7,0.01; 0.7,0.01;...
    0.7,0.01; 0.7,0.01; 0.7,0.01];
ub = [inf,2.5; inf,2.5; inf,2.5; ...
    inf,2.5; inf,2.5; inf,2.5];
for i = 1:6
    t = result.t_rst{i};
    s = result.s_gt{i};
    lg_s = log10(s);
    [res, gof] = fit(t',lg_s',weibull_type,...
        'Lower',lb(i,:),'Upper',ub(i,:));
    lss{i} = res;
    gofs{i} = gof;
    subplot(2,3,i);
    plot(t, lg_s, 'o'); hold on;
    tt = 0:t(end);
    plot(tt, res(tt));
    title(gof.rsquare)
    aphs(i) = res.aph;
    betas(i) = res.beta;
end
%% fit prep
ta = [70 90 80 110 70 70];
rsm = [0.1 0.1 0.2 0.2 0.2 0.2];
sp = [0.75 0.75 0.75 0.75 0.45 0.45];
vd = [2 2 2 2 1 2];
X = [ones(6,1) ta' sp' rsm' vd'];
%% fit aph
lg_aphs = log10(aphs);
sol_aphs = rref([X'*X X'*lg_aphs']);
sol_aphs = sol_aphs(:,end)
pred_lg_aphs = X*sol_aphs;
plot(lg_aphs,pred_lg_aphs,'o');
pred_aphs = 10.^pred_lg_aphs;
%% fit beta
sol_betas = rref([X'*X X'*betas']);
sol_betas = sol_betas(:,end)
pred_betas = X*sol_betas;
plot(betas,pred_betas,'o');
hold on; plot([1.8,2.5],[1.8,2.5]);
%% fit acc
figure(1);
for i = 1:6
    t = result.t_rst{i}';
    tt = (0:t(end))';
    s = result.s_gt{i}';
    lg_s = log10(s);
    pred_lgss = -1/2.303*(tt/pred_aphs(i)).^pred_betas(i);
    r2_lg = rsquare(lg_s,pred_lgss(t+1));
    fit_lg_acc(i) = r2_lg;
    subplot(2,3,i);
    plot(t,lg_s,'o');hold on;
    plot(tt,pred_lgss);
    title(num2str(r2_lg,3));
    r2_nm = rsquare(s, 10.^pred_lgss(t+1));
    fit_nm_acc(i) = r2_nm;
end
format longe;
fprintf('r2_fit_nm=%.4f \nr2_fit_lg=%.4f \n',...
    mean(fit_nm_acc),mean(fit_lg_acc));
%% test acc
figure(2);
ta_test = [110;90;70]; rsm_test = [0.1;0.2;0.2];
sp_test =[0.75;0.75;1]; vd_test = [2;2;2];
X_test = [ones(3,1),ta_test,sp_test,rsm_test,vd_test];
pred_aphs_test = 10.^(X_test*sol_aphs);
pred_betas_test = X_test*sol_betas;
for i = 1:3
    t = result.t_test{i}';
    tt = (0:t(end))';
    s = result.s_test{i}';
    lg_s = log10(s);
    pred_lgss = -1/2.303*(tt/pred_aphs_test(i)).^pred_betas_test(i);
    r2_lg = rsquare(lg_s,pred_lgss(t+1));
    r2_lg = rsquare(lg_s,pred_lgss(t+1));
    test_lg_acc(i) = r2_lg;
    subplot(1,3,i);
    plot(t,lg_s,'o');hold on;
    plot(tt,pred_lgss);
    title(num2str(r2_lg,3));
    r2_nm = rsquare(s, 10.^pred_lgss(t+1));
    test_nm_acc(i) = r2_nm;
end
fprintf('r2_test_nm=%.4f \nr2_test_lg=%.4f \n',...
    mean(test_nm_acc),mean(test_lg_acc));