data = xlsread(...
    'E:\\python_code\\Neural_Network\\probiotic\\excel\\itp_ft_s.xlsx', ...
    'Sheet3');
ori_time = data(:,1); T = data(:,2); X = data(:,3); s = data(:,end-1);
tag = logical(data(:,end)); s = s(tag); time = ori_time;
legend_size = 21; text_size = 23; tick_size = 18;
%% input
close;
fig1 = figure(1);
fig1.Position = [465.8000 129 729.6000 360.8000];
blue = [0,114,189]/255;
orange = [1,0,0];
h1 = axes('position', [0.15 0.2 0.6 0.7]); hold on;
plot(ori_time,time,'k>','markersize',3,'MarkerFaceColor','k');
plot(ori_time,T,'s','color',orange,'markersize',4,'MarkerFaceColor',orange);
plot(ori_time,X,'h','color',blue,'markersize',3,'MarkerFaceColor',blue);
xlim([0,120]);
% ylim([-1.3,2]);
hold off;h_leg=legend({'$\tilde{t}$','$\tilde{T}_\mathrm{d}$', ...
    '$\tilde{X}$'},'Position', [0.3144 0.2249 0.2976 0.1105], ...
    'fontsize',legend_size, 'Orientation', 'horizontal');
set(h_leg,'interpret','latex');
h1.FontName = 'Times New Roman';h1.FontSize=tick_size;
xlabel('Time (s)','FontName','Times New Roman','FontSize',text_size);
ylabel('$\tilde{t}$','Interpreter','latex','FontSize',text_size);
% SET h2
h2 = axes('position', [0.75 0.2 0.0 0.7]);
h2.YLim = h1.YLim;
h2.YTick = h1.YTick;
h2.YTickLabel=h1.YTickLabel;

set(h2, 'ycolor',orange, 'yaxislocation', 'right','xtick',[]);
hold on;
limX2 = get(h2, 'Xlim');
limY2 = get(h2, 'Ylim');
plot([limX2(2), limX2(2)], limY2, 'color',orange);
hold off;
h2.FontName = 'Times New Roman';h2.FontSize=tick_size;
ylabel('$\tilde{T_\mathrm{d}}$','Interpreter','latex','FontSize',text_size);
% SET h3
h3 = axes('position', [0.88 0.2 0.0 0.7]);
h3.YLim = h1.YLim;
h3.YTick = h1.YTick;
h3.YTickLabel=h1.YTickLabel;
set(h3, 'ycolor',blue, 'yaxislocation', 'right','xtick',[]);
hold on;
limX3 = get(h3, 'Xlim');
limY3 = get(h3, 'Ylim');
plot([limX3(2), limX3(2)], limY3, 'color',blue);
hold off;
h3.FontName = 'Times New Roman';h3.FontSize=tick_size;
ylabel('$\tilde{X}$','Interpreter','latex','FontSize',text_size);
% set(gcf,'color','none');
% set(gca,'color','none');
%% output
% close;
green = [0,19,18]/255;
fig2 = figure(2);
fig2.Position = [544.2000 280.2000 568 360.8000];
h4 = axes(); hold on;
plot(ori_time(tag),s(tag),'v','color',green,'markersize',9,'MarkerFaceColor',green); % ground truth
plot(ori_time(~tag), s(~tag), '.','Color',green,'MarkerSize',10); % itp label
h4_leg = legend({'$\tilde{s}^{\mathrm{grd}}$','$\tilde{s}^{\mathrm{itp}}$'}, ...
    'FontSize',legend_size,'Location','southwest');
set(h4_leg,'Interpret','latex');
h4.FontName = 'Times New Roman';h4.FontSize=tick_size;
xlabel('Time (s)','FontName','Times New Roman','FontSize',text_size);
ylabel('$\tilde{s}$','Interpreter','latex','FontSize',text_size);
% ylim([0,1.1]);
set(h4, 'ycolor',green,'yaxislocation', 'left');
% set(gca,'color','none');
% set(gcf,'color','none');