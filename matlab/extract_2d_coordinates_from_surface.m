function [ xy_elecs, im ] = extract_2d_coordinates_from_surface(triangles,vertices,xyz_elec,az,el,camzoom_ratio,angle_light,f_width,f_height)
% Make a surface brain plot and retrieve electrode locations in 2-D
% coordinates at a specific camera angle. Returns an image of the brain
% surface at that angle with a red background (so that it can be easily
% removed later). Also returns a list of 2-D points for the location of
% each electrode in this image. This effectively turns a 3-D representation
% of the brain + electrodes into a 2-D image for future plotting.
%
% Parameters
% ----------
% triangles : an M x 3 matrix of triangles composing the brain surface (see
%   trisurf for usage
% vertices : an M x 3 matrix specifying xyz vertices of the brain surface
%   (see trisurf for usage)
% xyz_elec : an M x 3 matrix specifying xyz positions of electrodes
% az : float, the azimuth of the camera (see `view`)
% el : float, the elevation of the camera (see `view`)
% camzoom_ratio : float, the amount to zoom in (1 is no zoom)
% angle_light : a length 2 vector specifying the angle of light
% f_width : float, the width of the figure to be created (in pixels)
% f_height : float, the height of the figure to be created (in pixels)
%
% Returns
% -------
% xy_elecs : N x 2 matrix, the xy positions of each electrode on the output
%   image
% im : image matrix, the image corresponding to the view of the brain
%   surface specified in the parameters.
vx = vertices(:, 1);
vy = vertices(:, 2);
vz = vertices(:, 3);

ex = xyz_elec(:, 1);
ey = xyz_elec(:, 2);
ez = xyz_elec(:, 3);

marker_size = f_width * .05;

figure('Position', [100, 100, f_width, f_height], 'Color', 'None');
hold on;
colormap gray;

% First plot the image at the right angle / etc and get the frame
surf = trisurf(triangles, vx, vy, vz, 'EdgeColor','none','LineStyle','none', ...
        'FaceColor', [1, 1, 1], 'FaceLighting', 'gouraud');
lightangle(angle_light(1), angle_light(2));
view(az, el);
camzoom(camzoom_ratio);
axis off

% Then, iterate through each electrode, plotting in bright green
% For each electrode, mask the 2d image w/ whatever is green
% Calculate the x/y of the centroid of the mask and store it
set(surf, 'visible', 'off');
set(gca(), 'XLimMode', 'manual', 'YLimMode', 'manual', 'ZLimMode', 'manual');
set(gca, 'Color', 'r');
set(gcf, 'color', 'r');

xy_elecs = [];
for ii = 1:size(ex, 1)
    h(ii) = scatter3(ex(ii), ey(ii), ez(ii), marker_size, [0 1 0], 'filled');
    fr = getframe(gca);
    im = fr.cdata;
    a = repmat([0 255 0]', 1, size(im, 1), size(im, 2));
    b = permute(a, [2, 3, 1]);
    comp = im == b;
    msk = all(comp, 3);
    [y, x] = find(msk);
    y = mean(y);
    x = mean(x);
    xy_elecs(ii, 1) = x;
    xy_elecs(ii, 2) = y;
    delete(h(ii));
end

set(surf, 'visible', 'on');
fr_brain = getframe(gca);
im = fr_brain.cdata;

scatter3(ex, ey, ez, marker_size, [0 1 0], 'filled');
title('Final brain plot + electrodes');
