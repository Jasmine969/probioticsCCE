%% aph
t = linspace(0,100,500);
aph = [0.1, 0.5, 1, 2, 3];
beta = 0.5;
figure(1);hold on;
for i = 1:length(aph)
    y = -1/2.303*(t/aph(i)).^beta;
    plot(t,y);
    labels{i} = num2str(aph(i));
end
legend(labels,'location','southwest');
title('alpha')
%% beta
% t = linspace(0,100,500);
% aph = 1;
% beta = [0.1,0.2,0.3,0.4,0.5];
% figure(2);hold on;
% for i = 1:length(beta)
%     y = -1/2.303*(t/aph).^beta(i);
%     plot(t,y);
%     labels{i} = num2str(beta(i));
% end
% legend(labels,'location','southwest');
% title('beta')