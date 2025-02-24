clc; clear all; close all;
%%
path = 'results\cylinder_1source_half_50st_12s\';

im = im2double(rgb2gray(imread([path '001.png'])));
full = zeros(size(im, 1), 360);

for i = 1:360
    im = im2double(rgb2gray(imread([path num2str(i,'%03.f') '.png'])));
    full(:, i) = im(:, round(size(im,2)/2));
end

% figure
% imshow(full);

%%
center_int = full(round(size(full,1)/2), :);
figure
plot(center_int);

[min_int, min_int_ind] = min(center_int);

center_int_new = zeros(size(center_int));
full_new = zeros(size(full));

for i=1:360
    ind = mod(min_int_ind + i, 360);
    if ind == 0
        ind = 360;
    end
    center_int_new(i) = center_int(ind);
    full_new(:, i) = full(:, ind);
end

figure
plot(center_int_new);

figure
imagesc(full_new);
colormap(gray);


%%
theta = linspace(0, 2*pi, 360);
figure
polarplot(theta,center_int_new)



