clear all; clc; close all;
fprintf('Start at %s\n', datestr(now,'HH:MM:SS.FFF'))
catalogue = {
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d1_h1_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d1_h2_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d1_h3_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d1_h4_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d1_h5_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d1_h6_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d1_h7_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d1_h8_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d1_h9_50st_20s\tempreco',...
        'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d2_h1_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d2_h2_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d2_h3_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d2_h4_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d2_h5_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d2_h6_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d2_h7_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d2_h8_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d2_h9_50st_20s\tempreco',...
        'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d3_h1_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d3_h2_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d3_h3_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d3_h4_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d3_h5_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d3_h6_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d3_h7_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d3_h8_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d3_h9_50st_20s\tempreco',...
'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d4_h1_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d4_h2_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d4_h3_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d4_h4_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d4_h5_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d4_h6_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d4_h7_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d4_h8_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d4_h9_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d5_h1_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d5_h2_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d5_h3_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d5_h4_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d5_h5_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d5_h6_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d5_h7_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d5_h8_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d5_h9_50st_20s\tempreco',...  
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d6_h1_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d6_h2_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d6_h3_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d6_h4_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d6_h5_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d6_h6_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d6_h7_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d6_h8_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d6_h9_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d7_h1_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d7_h2_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d7_h3_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d7_h4_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d7_h5_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d7_h6_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d7_h7_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d7_h8_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d7_h9_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d8_h1_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d8_h2_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d8_h3_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d8_h4_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d8_h5_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d8_h6_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d8_h7_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d8_h8_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d8_h9_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d9_h1_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d9_h2_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d9_h3_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d9_h4_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d9_h5_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d9_h6_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d9_h7_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d9_h8_50st_20s\tempreco',...
    'C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\thermal_model_3D\results\20s\cylinder_s1_d9_h9_50st_20s\tempreco'};


for i = 55: 63%numel(catalogue)
    A = dir([catalogue{i} '\*.mat']); % png
    no_images = length(A);
    
    if(numel(A)>1)
        disp(catalogue{i});
    else
        load([A(1).folder '\' A(1).name]);
        splitit = split(catalogue{i}, '\');
        name_splitit = splitit(end-1);
        name2save = [name_splitit{1} '.mat'];
        path2save = ['C:\Users\agata\Documents\_POLSL\_IR3D\_IR3D_modele_new\ir3d\reco_fin\20s\' name2save];
        save(path2save, 'ir3d');
    end

end

%%

% for i = 1: numel(catalogue)
%     A = dir([catalogue{i} '\*.mat']); % png
%     no_images = length(A);
%     
%     if(numel(A)>1)
%         A1 = A(1).date;
%         A2 = A(2).date;
%         
%         if(str2num(A1) >str2num(A2))
%             load([A(1).folder '\' A(1).name]);
%             splitit = split(catalogue{i}, '\');
%             name_splitit = splitit(end-1);
%             name2save = [name_splitit{1} '.mat'];
%             path2save = ['D:\_IR3D\reco_fin\20s\' name2save];
%             save(path2save, 'ir3d');
%         else
%             load([A(2).folder '\' A(2).name]);
%             splitit = split(catalogue{i}, '\');
%             name_splitit = splitit(end-1);
%             name2save = [name_splitit{1} '.mat'];
%             path2save = ['D:\_IR3D\reco_fin\20s\' name2save];
%             save(path2save, 'ir3d');
%         end
%     else
% %         load([A(1).folder '\' A(1).name]);
% %         splitit = split(catalogue{i}, '\');
% %         name_splitit = splitit(end-1);
% %         name2save = [name_splitit{1} '.mat'];
% %         path2save = ['D:\_IR3D\reco_fin\20s\' name2save];
% %         save(path2save, 'ir3d');
%     end
% 
% end