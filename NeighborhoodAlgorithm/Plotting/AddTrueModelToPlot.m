function [] = AddTrueModelToPlot (mTrue)

hax = flipud(get(gcf,'Children'));

for iax = 1:length(hax)
    axes(hax(iax));
    hold on;
    plot(mTrue(1), mTrue(2), 'k^', ...
        'MarkerSize', 10, 'Linewidth', 2);
    drawnow;
end

end