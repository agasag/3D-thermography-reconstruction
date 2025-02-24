function [BPI] = singleSliceRecoTemp(catalogue, A, no_images, no_slice, sizex, sizey)

for i=1:no_images
    image1=A(i).name;
    J = load([catalogue '\' image1]);
    J = J.I_temp(:,:,1);
    J = imresize(J, 0.3);
    minimalval = min(J(:));
    [x y] = size(J);
%     xval = abs(x-sizex); yval = abs(y-sizey); 
%     Jtemp = padarray(J, [xval/2 yval/2]);
      Jtemp = zeros(sizex, sizey); %Jtemp(J == 0) = 17;
      Jtemp(J>0) = J;
      Jtemp(Jtemp == 0) = minimalval;
%     Jtemp2 = [];
%     Jtemp2 = Jtemp(1:(end-1), 1:(end-1));
%     Jtemp = imresize(Jtemp, 0.2);
    sinogram(:,i) = Jtemp(no_slice,:,1); % !!!! 
end
    BPI = backprojectionReco(sinogram,0:1:359);
%   BPI = iradon(sinogram, 0:1:359);
 end