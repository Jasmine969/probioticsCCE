%% A
t = linspace(0,100,500);
As = -1:1:4;
B = -0.35;
C = -5;
figure(1);hold on;
for i = 1:length(As)
    y = C*exp(-exp(As(i)+B.*t))-C*exp(-exp(As(i)));
    plot(t,y);
    labels{i} = num2str(As(i));
end
legend(labels);
%% B
% clear;
% t = linspace(0,100,500);
% A = 2.36;
% C = -5;
% Bs = -2:0.5:-0.001;
% Bs = [Bs,-0.1];
% figure(2);hold on;
% for i = 1:length(Bs)
%     y = C*exp(-exp(A+Bs(i).*t))-C*exp(-exp(A));
%     plot(t,y);
%     labels{i} = num2str(Bs(i));
% end
% legend(labels);
%% C
% clear;
% t = linspace(0,100,500);
% A = 2.36;
% B = -1;
% Cs = -5:2:5;
% figure(3);hold on;
% for i = 1:length(Cs)
%     y = Cs(i)*exp(-exp(A+B.*t))-Cs(i)*exp(-exp(A));
%     plot(t,y);
%     labels{i} = num2str(Cs(i));
% end
% legend(labels);