function [] = get_projections(model, temp, angle_projection, path)
    %%
    range = 0;
    max_intensity = 255;
    
 	figure('units','normalized','outerposition',[0 0 1 1])
    while(max_intensity == 255)
        range = range + 0.01;
     	show_projection(model, temp, range);
        cropped = get_frame();
        I = cropped(20:(end-20), 20:(end-20));
   
        % if the colorbar is segmented than I == 0
        if(size(I, 2) < 200) 
            continue;
        end
        
        max_intensity = max(I(:));
        disp(max_intensity);
    end

    temperatures_map = linspace(min(temp), min(temp)+range, 256);
    %%
    axis vis3d
    steps = round(360/angle_projection);
    for i = 1:steps
%         disp(i);
        camorbit(angle_projection,0,'data',[0 0 1])
        delete(findobj(gca,'type','Text')); 
        delete(findobj(gca,'type','Quiver')); 
        drawnow
        
        I = get_frame();
        I_temp = temperatures_map(I+1);
        
        imwrite(I, [path '/img/', num2str(i,'%03.f'), '.png']);
        save([path '/temp/', num2str(i,'%03.f'), '.mat'], 'I_temp')
    end
end


