draw_point = @(p)scatter(p(1),p(2),70,'o','filled');
draw_line = @(p,q)plot([p(1),q(1)],[p(2),q(2)],'k--','LineWidth',2);
add_formula = @(x,y,f,sz)text(...
    x,y,f,'Interpreter','latex','FontSize',sz);
x = logspace(-3,0.1,100);
y = log10(x);
plot(x,y); grid on;hold on;axis equal;
axis([0.06,1,-0.8,0]);
cp = 1/log(10);
a = [cp,log10(cp)];
b = [0.7,log10(0.7)];
c = [0.27,log10(0.27)];
draw_point(a);
add_formula(0.41,-0.32,'$$A$$',15);
add_formula(0.1,-0.2,'$$x_A=\frac{1}{\ln{10}}$$',14);
dx = 0.05; Dx = 0.07;
a1 = [a(1)-dx,a(2)-dx];
a2 = [a(1)+dx,a(2)+dx];
a3 = [a2(1),a1(2)];
draw_line(a1,a2);draw_line(a1,a3);draw_line(a2,a3);
add_formula(0.41,-0.44,'$\Delta x$',14);
add_formula(0.5,-0.36,'$\Delta y$',14);
add_formula(0.6,-0.4,...
    '$$\lim\limits_{\Delta x\rightarrow 0}\frac{\Delta y}{\Delta x}=1$$'...
    ,14);

b1 = [b(1)-Dx,log10(b(1)-Dx)];
b2 = [b(1)+Dx,log10(b(1)+Dx)];
b3 = [b2(1),b1(2)];
draw_line(b1,b2);draw_line(b1,b3);draw_line(b2,b3);
add_formula(0.67,-0.26,'$\Delta x_1$',14);
add_formula(0.82,-0.16,'$\Delta y_1$',14);
add_formula(0.45,-0.15,...
    '$$\frac{\Delta y_1}{\Delta x_1}<1$$',14);

c1 = [c(1)-Dx,log10(c(1)-Dx)];
c2 = [c(1)+Dx,log10(c(1)+Dx)];
c3 = [c2(1),c1(2)];
draw_line(c1,c2);draw_line(c1,c3);draw_line(c2,c3);
add_formula(0.25,-0.74,'$\Delta x_2$',14);
add_formula(0.35,-0.6,'$\Delta y_2$',14);
add_formula(0.5,-0.65,...
    '$$\frac{\Delta y_2}{\Delta x_2}>1$$',14);
ax = gca;ax.FontName = 'Times New Roman';
ax.FontSize = 13;
xlabel('$x$','Interpreter','latex','FontSize',17);
ylabel('$\lg{x}$','Interpreter','latex','FontSize',17);
set(gcf,'color','none');