function BPI = backprojectionReco(sinogram,theta)

numOfParProj = size(sinogram,1);    % number of projections
theta = (pi/180)*theta;             % degrees -> radians

BPI = zeros(numOfParProj,numOfParProj);
midindex = floor(numOfParProj/2) + 1;   % middle of image
[xCoords,yCoords] = meshgrid(ceil(-numOfParProj/2):ceil(numOfParProj/2-1));

for i = 1:length(theta)
    rotCoords = round(midindex + xCoords*sin(theta(i)) + yCoords*cos(theta(i)));
    indices   = find((rotCoords > 0) & (rotCoords <= numOfParProj));
    newCoords = rotCoords(indices);
    BPI(indices) = BPI(indices) + sinogram(newCoords,i)./(length(theta)); % summation
end
