function show_projection(model, temp, range)
    pdeplot3D(model,'ColorMapData', temp)
    colormap('gray')
    caxis([min(temp) min(temp)+range])
    view(gca,[0,0])
    delete(findobj(gca,'type','Text')); 
    delete(findobj(gca,'type','Quiver')); 
end

