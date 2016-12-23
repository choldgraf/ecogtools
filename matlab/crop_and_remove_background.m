function [im_crop, xy_crop] = crop_and_remove_background(im, xy)

greys = mean(im, 3);
rows = mean(greys, 2);
columns = mean(greys, 1);

% X limits
ixs_x = find(columns ~= 85);
ixs_y = find(rows ~= 85);

xmin = min(ixs_x);
xmax = max(ixs_x);
ymin = min(ixs_y);
ymax = max(ixs_y);

im_crop = im(ymin:ymax, xmin:xmax, 1:end);
greys_crop = mean(im_crop, 3);

% Change the xy values
xy_crop = xy;
xy_crop(1:end, 1) = xy_crop(1:end, 1) - xmin;
xy_crop(1:end, 2) = xy_crop(1:end, 2) - ymin;


% Remove background
mask = greys_crop == 85;
mask = repmat(mask, [1, 1, 3]);
im_crop(mask) = 0;