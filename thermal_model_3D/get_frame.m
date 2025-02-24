function [cropped] = get_frame()
    F = getframe(gcf);
    [Irgb, ~] = frame2im(F);
    I = im2double(rgb2gray(Irgb));
    mask = I < 1;
    BW2 = bwareafilt(mask, 1);
    measurements = regionprops(BW2, 'BoundingBox');

    cropped = rgb2gray(imcrop(Irgb, measurements.BoundingBox));
end

