%% load dataset
load('other_model_dataset.mat', 'result');
gomp = fittype('C*exp(-exp(A+B*t))-C*exp(-exp(A))',...
    'dependent','y','independent','t');
%% fit time-s
for i = 1:6
    t = result.t_rst{i}';
    s = result.s_gt{i}';
    lg_s = log10(s);
    lb = [0,-Inf,-Inf];ub=[Inf,0,0];
    if i == 1
        lb = [1 -Inf -10];ub=[Inf,0,0];
    elseif i == 4
        lb = [1 -0.1 -10];ub=[10,0,-1];
    elseif i == 5
        lb = [1 -1 -5];ub=[50,0,-1.5];
    elseif i==6
        lb = [1 -0.05 -Inf]; ub=[Inf,0,-1.5];
    end
    [fitres,gof] = fit(t,lg_s,gomp,'Lower',lb,'Upper',ub);
    gps{i} = fitres;
    gofs{i} = gof;
    As(i) = fitres.A;Bs(i) = fitres.B;Cs(i) = fitres.C;
end
%% fit prep
ta = [70 90 80 110 70 70];
sp = [0.75 0.75 0.75 0.75 0.45 0.45];
rsm = [0.1 0.1 0.2 0.2 0.2 0.2];
vd = [2 2 2 2 1 2];
X = [ones(6,1) ta' sp' rsm' vd'];
%% fit A
sol_A = rref([X'*X X'*As']);
sol_A = sol_A(:,end)
pred_A = X*sol_A;
% plot(As,pred_A,'o');hold on;plot([0,10],[0,10]);
%% fit B
sol_B = rref([X'*X X'*Bs']);
sol_B = sol_B(:,end)
pred_B = X*sol_B;
% plot(Bs,pred_B,'o');
% hold on;plot([0,-0.1],[0,-0.1]);
%% fit C
sol_C = rref([X'*X X'*Cs']);
sol_C = sol_C(:,end)
pred_C = X*sol_C;
% plot(Cs,pred_C,'o');hold on;plot([-8,0],[-8,0]);
%% fit acc
figure(1);
for i = 1:6
    t = result.t_rst{i}';
    tt = (0:t(end))';
    s = result.s_gt{i}';
    lg_s = log10(s);
    pred_lgss = pred_C(i)*exp(-exp(pred_A(i)+pred_B(i)*tt))...
        -pred_C(i)*exp(-exp(pred_A(i)));
    r2_lg = rsquare(lg_s,pred_lgss(t+1));
    fit_lg_acc(i) = r2_lg;
    subplot(2,3,i);
    plot(t,lg_s,'o');hold on;
    plot(tt,pred_lgss);
    title(num2str(r2_lg,3));
    r2_nm = rsquare(s, 10.^pred_lgss(t+1));
    fit_nm_acc(i) = r2_nm;
end
fprintf('r2_fit_nm=%.4f \nr2_fit_lg=%.4f \n',...
    mean(fit_nm_acc),mean(fit_lg_acc));
%% test acc
figure(2);
ta_test = [110;90;70]; rsm_test = [0.1;0.2;0.2];
sp_test =[0.75;0.75;1]; vd_test = [2;2;2];
X_test = [ones(3,1),ta_test,sp_test,rsm_test,vd_test];
pred_A_test = X_test*sol_A;
pred_B_test = X_test*sol_B;
pred_C_test = X_test*sol_C;
for i = 1:3
    t = result.t_test{i}';
    tt = (0:t(end))';
    s = result.s_test{i}';
    lg_s = log10(s);
    pred_lgss = Cs(i)*exp(-exp(pred_A_test(i)+pred_B_test(i)*tt))...
        -pred_C_test(i)*exp(-exp(pred_A_test(i)));
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