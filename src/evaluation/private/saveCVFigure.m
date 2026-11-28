function saveCVFigure(fig, directory, name)
exportgraphics(fig, fullfile(directory, name + ".png"), 'Resolution', 200);
exportgraphics(fig, fullfile(directory, name + ".pdf"), 'ContentType', 'vector');
end
