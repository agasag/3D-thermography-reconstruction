clc; clear all; close all;
%%
model_name = 'cylinder_s1_d5';
model_path = 'models/';

time_stop = 5; % seconds
time_step = 5; % seconds

temp_range = 1; % for visualisation: from min_temp to min_temp + temp_range
temp_source = 30;

show_all = false;

save_projections = true;
time_projection = 5; % seconds
angle_projection = 1;
projections_path = ['results/' model_name '_' ...
                    num2str(temp_source) 'st_' ...
                    num2str(time_projection) 's/'];


model_path = [model_path '/' model_name '.stl'];

temp_interpolation = true;
resolution = [100, 200]; % points in x and y

%%
model = createpde('thermal','transient');
importGeometry(model, model_path);

figure
pdegplot(model,'CellLabels','on', 'FaceLabels', 'on', 'FaceAlpha',0.2);
view([-5 -47])
%%
generateMesh(model, 'HMax', 0.01);
figure
pdemesh(model, 'FaceAlpha',0.2)
view([0,90])
%%
% properties of muscle tissue
thermalProperties(model,'ThermalConductivity',0.49, ...
                        'MassDensity',1090, ...
                        'SpecificHeat',3.8);
% air, free convection
% thermalBC(model,'Face',1:model.Geometry.NumFaces, ...
%                 'ConvectionCoefficient',10, ...
%                 'AmbientTemperature',20);

thermalIC(model, 20);
thermalBC(model, 'Face', 1, 'Temperature', temp_source);
% thermalBC(model, 'Face', 2, 'Temperature', temp_source);
% thermalBC(model, 'Face', 3, 'Temperature', temp_source);

tlist =0:time_step:time_stop;
R1 = solve(model,tlist);
%% 
if show_all
    for i = 1:length(tlist)
        temp = R1.Temperature(:,i);
        figure('units','normalized','outerposition',[0 0 1 1])
        pdeplot3D(model,'ColorMapData', temp)
        colormap('gray')
        caxis([min(temp) min(temp)+temp_range])
    end
end

%%
if save_projections
    if ~exist(projections_path, 'dir')
       mkdir(projections_path);
       mkdir([projections_path '/img/']);
       mkdir([projections_path '/temp/']);
    end
    ind = find(tlist == time_projection);
    temp = R1.Temperature(:, ind);
    
    get_projections(model, temp, angle_projection, projections_path);
end
%%
if(temp_interpolation)
    for res = resolution
         interp_path = [projections_path '/interpolation/'];
         if ~exist(interp_path, 'dir')
           mkdir(interp_path);
         end

        cord_min = min(model.Mesh.Nodes, [], 2);
        cord_max = max(model.Mesh.Nodes, [], 2);

        x = linspace(cord_min(1), cord_max(1), res);
        y = linspace(cord_min(2), cord_max(2), res);
        z = linspace(cord_min(3), cord_max(3), res * ...
                                               (cord_max(3) / cord_max(1)));

        [X,Y,Z] = meshgrid(x, ...
                           y, ...
                           z);

        Tintrp = interpolateTemperature(R1, X, Y, Z, find(tlist==time_projection));
        Tintrp_mesh = reshape(Tintrp,size(X));

        save([interp_path, 'temperature_inside', num2str(res), '.mat'], 'Tintrp_mesh');
    end
end