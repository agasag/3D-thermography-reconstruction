clc; clear all; close all;
%%
im = im2double(rgb2gray(imread('image.png')));

cent_profile = im(:, size(im, 2)/2);

[min_c, ind_min] = min(cent_profile);
[max_c, ind_max] = max(cent_profile);

v1 = [ind_min, min_c];
v2 = [ind_max, max_c];



dist = zeros(1, length(cent_profile));
for i = 1:length(cent_profile)
    dist(i) = point_to_line([i, cent_profile(i), 0], [v1, 0], [v2, 0]); 
end

marg = 10;

[max_dist, ind_max_dist] = max(dist(marg:end-marg));
figure
plot(dist);
hold on
plot(ind_max_dist + marg, max_dist, 'or');

figure
plot(cent_profile);
hold on
plot([ind_min, ind_max], [min_c, max_c]);
plot(ind_max_dist + marg, cent_profile(ind_max_dist + marg), 'or');

%%
[val, ind] = unique(cent_profile, 'legacy');

figure
plot(ind, val);
hold on
plot([ind(1), ind(end)], [val(1), val(end)]);
%%
