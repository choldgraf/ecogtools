addpath('../);

surf = load('./surface.mat');

elecs = csvread('./elec_coords_3d.csv');


%% Collect data
surf = surf.surface;
xyz = elecs(1:end, 2:end);
%% First viz
ax = plot_3d_brain(surf.tri, surf.pos, xyz);

%% Position the camera correctly, then set the correct zoom here
spin = -25;
camroll(spin);

%% Get the light angle right
light_angle = [45, 45];
lightangle(light_angle(1), light_angle(2));

%% Then take the azimuth/elevation
[az, el] = view;

%% Now create the options parameter
opt.az = az;
opt.el = el;
opt.spin = spin;
opt.figsize = [800, 800];
opt.angle_light = light_angle;
opt.camzoom_ratio = .8;

%% do the extracting
[xy, im] = extract_2d_coordinates_from_surface(surf.tri, surf.pos, xyz, opt);

%% Plot the result
close all;
figure
imshow(im)
hold on

scatter(xy(1:end, 1), xy(1:end, 2))

%% Now strip the background and clip
[im_crop, xy_crop] = crop_and_remove_background(im, xy);


%% Show cropped image
close all;
figure
imshow(im_crop)
hold on
scatter(xy_crop(1:end, 1), xy_crop(1:end, 2))
%% Save positions and image
imwrite(im_crop, './brain.png')
csvwrite('./elecs_xy.csv', xy_crop);
