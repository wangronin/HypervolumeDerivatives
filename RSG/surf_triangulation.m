%%%-----------------------V1.3.4 01/07/2024-----------------------------%%%
function DT = surf_triangulation(Py, clean_method, threshold, comp_name, show_plots, save_plots )
num_obj = size(Py,2);

%--------------------------------------------------------%
%--------------------k>3 FILLING-------------------------%
%--------------------------------------------------------%
        %CHANGE OF COORDINATES
    normal = shift_vect(Py,show_plots);
%     normal = [1,1,1]; % ONLY FOR CONV3.4
    
    if show_plots
        %Plot THE SHIFT DIRECTION
        figure
        s1 = scatter3(Py(:,1),Py(:,2),Py(:,3),'k','.','DisplayName', 'Py');
        hold on
        s2 = scatter3(Py(:,1)+normal(1),Py(:,2)+normal(2),Py(:,3)+normal(3),'y','.','DisplayName', 'Py + \eta');
        for i=1:size(Py,1)
            plot3([Py(i,1),Py(i,1)+normal(1)],[Py(i,2),Py(i,2)+normal(2)],[Py(i,3),Py(i,3)+normal(3)])
        end
        legend([s1,s2])
        title('Shift direction \eta')
        figSettings
        if save_plots
            saveas(gcf, [ pwd '/PLOTS/' comp_name '_shift_direction.fig' ] );
            saveas(gcf, [ pwd '/PLOTS/' comp_name '_shift_direction.pdf' ] );
            saveas(gcf, [ pwd '/PLOTS/' comp_name '_shift_direction.png' ] );
        end
    end

    
    if size(normal,1) == 1
        [Q,R] = qr(normal');
    else
        [Q,R] = qr(normal);
    end
    Q = sign(R(1,1))*Q;
    Ry = Py*Q';
    
    if num_obj==3
        DT_raw = delaunay(Ry(:,2),Ry(:,3));
    else
        DT_raw = delaunay(Ry(:,2),Ry(:,3),Ry(:,4));
    end
    
    if show_plots
        %Projection:
        figure
        if num_obj==3
            scatter(Ry(:,2),Ry(:,3),'.');
        else
            scatter3(Ry(:,2),Ry(:,3),Ry(:,4),'.');
        end
        title('Projection')
        figSettings
        if save_plots
            saveas(gcf, [ pwd '/PLOTS/' comp_name '_projection.fig' ] );
            saveas(gcf, [ pwd '/PLOTS/' comp_name '_projection.pdf' ] );
            saveas(gcf, [ pwd '/PLOTS/' comp_name '_projection.png' ] );
        end
    
        %Projection Triangulation:
        figure
        if num_obj==3
            triplot(DT_raw,Ry(:,2),Ry(:,3));
        else
            trisurf(DT_raw,Ry(:,2),Ry(:,3),Ry(:,4));
            zlabel('f_3')
        end
        xlabel('f_1')
        ylabel('f_2')
        title('Projection Raw Triangulation')
        figSettings
        if save_plots
            saveas(gcf, [ pwd '/PLOTS/' comp_name '_projection_triangulation.fig' ] );
            saveas(gcf, [ pwd '/PLOTS/' comp_name '_projection_triangulation.pdf' ] );
            saveas(gcf, [ pwd '/PLOTS/' comp_name '_projection_triangulation.png' ] );
        end
    
        %Whole Triangulation:
        if num_obj==3
            figure
            trisurf(DT_raw,Ry(:,1),Ry(:,2),Ry(:,3));
            xlabel('f_1')
            ylabel('f_2')
            zlabel('f_3')
        else
            figure
            trisurf(DT_raw,Ry(:,1),Ry(:,2),Ry(:,3));
            xlabel('f_1')
            ylabel('f_2')
            zlabel('f_3')
            figure
            trisurf(DT_raw,Ry(:,1),Ry(:,2),Ry(:,4));
            xlabel('f_1')
            ylabel('f_2')
            zlabel('f_4')
            figure
            trisurf(DT_raw,Ry(:,1),Ry(:,3),Ry(:,4));
            xlabel('f_1')
            xlabel('f_3')
            xlabel('f_4')
            figure
            trisurf(DT_raw,Ry(:,2),Ry(:,3),Ry(:,4));
            xlabel('f_2')
            xlabel('f_3')
            xlabel('f_4')
        end
        title('Raw Triangulation')
        figSettings
        if save_plots
            saveas(gcf, [ pwd '/PLOTS/' comp_name '_triangulation.fig' ] );
            saveas(gcf, [ pwd '/PLOTS/' comp_name '_triangulation.pdf' ] );
            saveas(gcf, [ pwd '/PLOTS/' comp_name '_triangulation.png' ] );
        end
    end

    if strcmp(clean_method,'cond')
        DT = rmtriangle( DT_raw, Ry(:,1:end), @tri_cond, threshold, comp_name, show_plots, save_plots ); % conditional number
    elseif strcmp(clean_method,'area')
        DT = rmtriangle( DT_raw, Ry(:,1:end), @tri_area, threshold, comp_name, show_plots, save_plots ); % Area
    elseif strcmp(clean_method,'off')
        DT = DT_raw;
        disp('NO CLEANING');
    else %'long'
        DT = rmtriangle( DT_raw, Ry(:,1:end), @tri_long, threshold, comp_name, show_plots, save_plots ); % longitud
    end

    if show_plots
        %Projection Triangulation Clean:
        figure
        if num_obj==3
            triplot(DT,Ry(:,2),Ry(:,3));
        else
            trisurf(DT,Ry(:,2),Ry(:,3),Ry(:,4));
            zlabel('f_3')
        end
        xlabel('f_1')
        ylabel('f_2')
        title('Projection Triangulation Clean')
        figSettings
        if save_plots
            saveas(gcf, [ pwd '/PLOTS/' comp_name '_projection_triangulation_clean.fig' ] );
            saveas(gcf, [ pwd '/PLOTS/' comp_name '_projection_triangulation_clean.pdf' ] );
            saveas(gcf, [ pwd '/PLOTS/' comp_name '_projection_triangulation_clean.png' ] );
        end
    
        %Clean Triangulation:
        if num_obj == 3
            figure
            trisurf(DT,Py(:,1),Py(:,2),Py(:,3));
            xlabel('f_1')
            ylabel('f_2')
            zlabel('f_3')
        else
            figure
            trisurf(DT,Py(:,1),Py(:,2),Py(:,3));
            xlabel('f_1')
            ylabel('f_2')
            zlabel('f_3')
            figure
            trisurf(DT,Py(:,1),Py(:,2),Py(:,4));
            xlabel('f_1')
            ylabel('f_2')
            zlabel('f_4')
            figure
            trisurf(DT,Py(:,1),Py(:,3),Py(:,4));
            xlabel('f_1')
            xlabel('f_3')
            xlabel('f_4')
            figure
            trisurf(DT,Py(:,2),Py(:,3),Py(:,4));
            xlabel('f_2')
            xlabel('f_3')
            xlabel('f_4')
        end
        title('Clean Triangulation')
        figSettings
        if save_plots
            saveas(gcf, [ pwd '/PLOTS/' comp_name '_triangulation_clean.fig' ] );
            saveas(gcf, [ pwd '/PLOTS/' comp_name '_triangulation_clean.pdf' ] );
            saveas(gcf, [ pwd '/PLOTS/' comp_name '_triangulation_clean.png' ] );
        end
    end

end

%---------------------------FUNCTIONS-------------------------------------%

%--Triangulation Cleaning----%
function Volume = tri_area( tridots ) 
    num_obj = size(tridots,2);
    %-----Compute Volume-------------%
    Volume = zeros(num_obj,num_obj-1);
    for l=2:num_obj
        Volume(:,l-1) = tridots(l,:) - tridots(1,:);
    end
    Volume = det(Volume'*Volume)/factorial(num_obj);
    %-------------------------------------%
end

function kappa = tri_cond( tridots ) 
    kappa = cond(tridots);
end

function A = tri_long( tridots ) 
    if size(tridots,2) == 3
        A = max([...
            norm( tridots(1,:) - tridots(2,:) ),...
            norm( tridots(1,:) - tridots(3,:) ),...
            norm( tridots(2,:) - tridots(3,:) )...
            ]);
    else
        A = max([...
            norm( tridots(1,:) - tridots(2,:) ),...
            norm( tridots(1,:) - tridots(3,:) ),...
            norm( tridots(1,:) - tridots(4,:) ),...
            norm( tridots(2,:) - tridots(3,:) )...
            norm( tridots(2,:) - tridots(4,:) )...
            norm( tridots(3,:) - tridots(4,:) )...
            ]);
    end
end
%----------------------------%

%-------------------------------------------------------------------------%
