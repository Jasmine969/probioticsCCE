clear;
data_input = xlsread(...
    'E:\\python_code\\Neural_Network\\probiotic\\excel\\itp_ft_s.xlsx', ...
    'Sheet1');
data_output = xlsread(...
    'E:\\python_code\\Neural_Network\\probiotic\\excel\\raw_ft_s.xlsx', ...
    'Sheet1');
time = data_input(:,1); T = data_input(:,2); X = data_input(:,3);
s = data_output(:,end); time_output = data_output(:,1);
legend_size = 21; text_size = 23; tick_size = 17;
%% input
close;
fig1 = figure(1);
% ×ø±êÇø¿í¶È447.92
fig1.Position = [465.8000 129 760 360.8000];
scale_T = 1.1; scale_X = 1.1;
blue = [0,114,189]/255;
orange = [1,0,0];
f1 =figure(1);
h1 = axes('position', [0.12 0.2 0.58 0.7]); hold on;
real2ax = @(var,scale)((var-min(var))/range(var)*range(time)+min(time))*scale;
T_new = real2ax(T,scale_T); X_new = real2ax(X,scale_X);
plot(time,time,'ko','markersize',3);
plot(time,T_new,'s','color',orange,'markersize',4);
plot(time,X_new,'>','color',blue,'markersize',3);
hold off; h_leg = legend({'$t$','$T_\mathrm{d}$','$X$'},...
    'fontsize',legend_size, 'Position',[0.5434 0.2916 0.1092 0.2932]);
set(h_leg,'Interpret','latex')
h1.FontName = 'Times New Roman';h1.FontSize=tick_size;
xlabel('Time (s)','FontName','Times New Roman','FontSize',text_size);
ylabel('$t~(\mathrm{s})$','Interpret','latex','FontSize',text_size);
xlim([0,270]);
% SET h2
h2 = axes('position', [0.70 0.2 0.0 0.7]);
h2.YLim = h1.YLim;
h2.YTick = h1.YTick;
h2.YTickLabel=h1.YTickLabel;
for i=1:length(h1.YTickLabel)
    tmp = str2double(h1.YTickLabel(i));
    tmp = (tmp/scale_T - min(time))/range(time);
    h2.YTickLabel(i) = {num2str(tmp * range(T) + min(T),3)};
end
set(h2, 'ycolor',orange, 'yaxislocation', 'right','xtick',[]);
hold on;
limX2 = get(h2, 'Xlim');
limY2 = get(h2, 'Ylim');
plot([limX2(2), limX2(2)], limY2, 'color',orange);
hold off;
h2.FontName = 'Times New Roman';h2.FontSize=tick_size;
ylabel('$T_\mathrm{d}~(\mathrm{K})$','Interpret','latex','FontSize',text_size);
% SET h3
h3 = axes('position', [0.82 0.2 0.0 0.7]);
h3.YLim = h1.YLim;
h3.YTick = h1.YTick;
h3.YTickLabel=h1.YTickLabel;
for i=1:length(h1.YTickLabel)
    tmp = str2double(h1.YTickLabel(i));
    tmp = (tmp/scale_X - min(time))/range(time);
    h3.YTickLabel(i) = {num2str(tmp * range(X) + min(X),3)};
end
set(h3, 'ycolor',blue, 'yaxislocation', 'right','xtick',[]);
hold on;
limX3 = get(h3, 'Xlim');
limY3 = get(h3, 'Ylim');
plot([limX3(2), limX3(2)], limY3, 'color',blue);
hold off;
h3.FontName = 'Times New Roman';h3.FontSize=tick_size;
ylabel('$X~(\mathrm{kg/kg})$','Interpret','latex','FontSize',text_size);
%% output
close;
green = [0,19,18]/255;
fig2 = figure(2);
fig2.Position = [544.2000 280.2000 568 360.8000];
h4 = axes();
plot(time_output,s,'v','color',green,'markersize',7);
h4.FontName = 'Times New Roman';h4.FontSize=tick_size;
h_leg2 = legend({'$s^\mathrm{grd}$'},...
    'fontsize',legend_size, 'Position',[0.1645 0.2350 0.1653 0.1049]);
set(h_leg2,'Interpret','latex')
xlabel('Time (s)','FontName','Times New Roman','FontSize',text_size);
ylabel('$s$','Interpret','latex','FontSize',text_size);
set(h4, 'ycolor',green,'yaxislocation', 'left','box','off');