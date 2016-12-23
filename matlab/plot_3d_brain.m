function [ax] = plot_3d_brain(triangles,vertices,xyz_elec)
% Plot a 3d brain using trisurf.
vx = vertices(:, 1);
vy = vertices(:, 2);
vz = vertices(:, 3);

ex = xyz_elec(:, 1);
ey = xyz_elec(:, 2);
ez = xyz_elec(:, 3);

figure('Color', 'None');
hold on;
axis vis3d
colormap gray;

% First plot the image at the right angle / etc and get the frame
surf = trisurf(triangles, vx, vy, vz, 'EdgeColor','none','LineStyle','none', ...
        'FaceColor', [.5, .5, .5], 'FaceLighting', 'gouraud');
lightangle(45, 45);
axis off

scatter3(ex, ey, ez, 10, [0 1 0], 'filled');
title('Brain plot + electrodes');
ax = gca;
