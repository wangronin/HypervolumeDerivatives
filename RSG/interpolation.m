%%%-----------------------V1.4 16/OCT/2024-----------------------------%%%
% function [Fy] = interpolation( Py, Nf, clean_method, threshold, name, epsInterval, eps_def, minptsInterval, show_plots, save_plots, degeneration )
function [Fy] = interpolation( Py, Nf, clean_method, threshold, name, epsInterval, eps_def, minptsInterval, show_plots, save_plots)
    num_obj = size(Py,2);
    degeneration = false;

%--------------------------------------------------------%
%--------------------2D FILLING--------------------------%
%--------------------------------------------------------%
    if num_obj == 2
        %2D Filling
%         epsInterval = [0.25,0.25];
%         eps_def = 0.01;
%         minptsInterval = [2,3];
                
        [~,~,C,num_clusters,~,~] = cluster_gridsearch_(Py,epsInterval,eps_def,minptsInterval);

        if show_plots
            figure
            for i=1:num_clusters
                scatter(C{i}(:,1),C{i}(:,2),'filled');
                hold on
            end
            scatter(Py(:,1),Py(:,2),'.')
            figSettings
            xlabel('f_1')
            ylabel('f_2')
            title('Component Detection')
            if save_plots
                saveas(gcf, [ pwd '/PLOTS/' name '_Components.fig' ] );
                saveas(gcf, [ pwd '/PLOTS/' name '_Components.pdf' ] );
                saveas(gcf, [ pwd '/PLOTS/' name '_Components.png' ] );
            end
        end

        %------------------------------------------------INTERPOLATION
        I = {};
        %Modificación: Proporcional al número de puntos:
        TotalPointsCluster = 0;
        for i=1:num_clusters
            TotalPointsCluster = TotalPointsCluster + size(C{i},1);
        end
        %
        
        for i=1:num_clusters
            
            close all

%             I{i} = interpolate_tst(C{i},round(Nf/num_clusters));
            %Modificación: Proporcional al número de puntos:
            I{i} = interpolate_tst(C{i},round(Nf*size(C{i},1)/TotalPointsCluster));
        end
        comp_idx_tot = [];
        Fy = [];
        for i=1:length(I)
            for j= 1:length(I{i})
                comp_idx_tot(end+1) = i; 
                Fy(end+1,:) = I{i}(j,:);
            end
        end
        
%--------------------------------------------------------%
%--------------------k>3 FILLING-------------------------%
%--------------------------------------------------------%
    elseif num_obj >= 3
        %--------------------COMPONENT DETECTION--------------------
%         epsInterval = [0.25,0.30]; %TESTS!!! MODIFIED VALUE
%         eps_def = 0.05; 
%         minptsInterval = [3,4];

        [~,~,C,num_clusters,~,~] = cluster_gridsearch_(Py,epsInterval,eps_def,minptsInterval);

        if show_plots
            figure
            for i=1:num_clusters
                scatter3(C{i}(:,1),C{i}(:,2),C{i}(:,3),'filled');
                hold on
            end
            scatter3(Py(:,1),Py(:,2),Py(:,3),'.')
            figSettings
            xlabel('f_1')
            ylabel('f_2')
            zlabel('f_3')
            title('Component Detection')
            if save_plots
                saveas(gcf, [ pwd '/PLOTS/' name '_Components.fig' ] );
                saveas(gcf, [ pwd '/PLOTS/' name '_Components.pdf' ] );
                saveas(gcf, [ pwd '/PLOTS/' name '_Components.png' ] );
            end
        end

        %-----------------------Random INTERPOLATION--------------------
        %Modificación: Proporcional al número de puntos:
        TotalPointsCluster = 0;
        for i=1:num_clusters
            TotalPointsCluster = TotalPointsCluster + size(C{i},1);
        end
%             I{i} = interpolate_tst(C{i},round(Nf*size(C{i},1)/TotalPointsCluster));
        
        %RANDOM INTERPOLATION FOR EACH CLUSTER
        I_rand = cell(1,num_clusters);
        for i = 1:num_clusters
            
            close all

            %Modificación: Proporcional al número de puntos: (comentar para
            %todos mismo num interpol pts
            interpol_points = round(Nf*size(C{i},1)/TotalPointsCluster);
            
            if size(C{i},1) <= 3
                I_rand{i}=C{i};
            elseif size(unique(C{i}(:,1)),1) <= 3
                I_rand{i}=C{i};
            elseif degeneration
                %DEGENERATION CASE
                
            else
                comp_name = [name '_comp' int2str(i)];
        	    DT = surf_triangulation( C{i}, clean_method, threshold, comp_name, show_plots, save_plots );

                total_volume = 0;
                all_volumes = [];
                for j=1:size(DT,1)
                    %-----Compute Volume-------------%
                    Volume = zeros(num_obj,num_obj-1);
                    for l=2:num_obj
                        Volume(:,l-1) = C{i}(DT(j,l),:) - C{i}(DT(j,1),:);
                    end
                    Volume = det(Volume'*Volume)/factorial(num_obj);
                    all_volumes(end+1) = Volume;
                    total_volume = total_volume + Volume;
                    %-------------------------------------%
                end
%                 I_rand{i} = interpolate_triangle_random(DT.Points(K(1,1),:),DT.Points(K(1,2),:),DT.Points(K(1,3),:),total_area,interpol_points);
                I_rand{i} = interpolate_simplex_random(C{i}(DT(1,:),:),all_volumes(1),total_volume,interpol_points);
                for j=2:size(DT,1)
%                     points = interpolate_triangle_random(DT.Points(K(j,1),:),DT.Points(K(j,2),:),DT.Points(K(j,3),:),total_area,interpol_points);
                    points = interpolate_simplex_random(C{i}(DT(j,:),:),all_volumes(j),total_volume,interpol_points);
                    I_rand{i} = [I_rand{i};points];
                end
            end
        end
        
        %FOR FUNCTION OUTPUT
        I = I_rand;

        Fy = [];
        comp_idx_tot = [];
        for i=1:size(I_rand,2)
            for j= 1:size(I_rand{i},1)
                comp_idx_tot(end+1) = i;
                Fy(end+1,:) = I_rand{i}(j,:);
            end
        end
        
        %EndPoints
        [~,I] = max(Py);
        if length(unique(I)) ~= num_obj
            %SOLUCION MIN
            [~,I] = min(Py);
            if length(unique(I)) ~= num_obj
                %SOLUCION CLUSTER
                [~,~,~,~,I] = kmedoids(Py,num_obj,'Options',statset('MaxIter',10));%MODIFICACION
            end
        end
        for obj=1:num_obj
            Fy(end+1,:) = Py(I(obj),:);
        end
        %end EndPoints

        
    end
end


%---------------------------FUNCTIONS-------------------------------------%

%----2D Interpolation-----%
function I = interpolate_tst(X,Nf)

    [n,p] = size(X);
    I = zeros(Nf,p);
    Points = sortrows(X,1);
    endpoints = [Points(1,:); Points(end,:)];
    distances = zeros(1,n-1);
    points_p_segment = zeros(1,n-1);
    for i=1:n-1
        distances(i) = norm(Points(i+1,:)-Points(i,:));
    end
    % maxdistance = norm(Points(1,:),Points(end,:));
    total_length = sum(distances); %SUM OF THE LENGTH SEGMENTS
    cum_dist = cumsum(distances); %CumSum to see where to put points
    
%     if ~increase_interval
        interpol_length = total_length/(Nf-1);
        dist_left = zeros(1,n);
        for i=1:n-1
            ratio = (distances(i)+dist_left(i))/interpol_length;
            points_p_segment(i) = floor(ratio);
            dist_left(i+1) = (ratio-floor(ratio))*interpol_length;
        end
        count=1;
        for seg=1:n-1
            if points_p_segment(seg)>0
                direction = Points(seg+1,:)-Points(seg,:);
                direction = direction/norm(direction);
                I(count,:) = Points(seg,:)+(interpol_length-dist_left(seg))*direction;
                count=count+1;
                for i=2:points_p_segment(seg)
                    I(count,:) = I(count-1,:)+interpol_length*direction;
                    count=count+1;
                end
            end
        end
%     end 

    I(end-1:end,:) = endpoints;
    if count<Nf-2
        disp('CUIDADO, MENOS PUNTOS ENCONTRADOS, SET DEGENERADO')
        for i=count:Nf-2
            I(i,:) = endpoints(randi(2),:);
        end
    end
end
%-------------------------%

%----k>3 Interpolation----%
function points = interpolate_simplex_random(P,P_volume,total_volume,total_points)
num_obj = size(P,2);

tri_points = ceil(total_points*P_volume/total_volume); %points for triangle

points = zeros(tri_points,num_obj);
alpha = rand(tri_points,num_obj);
for i=1:tri_points
    alpha(i,:) = alpha(i,:)/norm(alpha(i,:));
    alpha(i,:) = alpha(i,:).^2;
    points(i,:) = sum(P.*alpha(i,:)');
%     points(i,:) = alpha(1,1)*P(1,:)+alpha(1,2)*P(2,:)+alpha(1,3)*P(3,:)+alpha(1,4)*P(4,:);
end

end
%-------------------------%

%--Component Detection----%
function [eps_fin,minpts_fin,C_fin,num_clust_fin,idx_fin,avrg_dist] = cluster_gridsearch_(M,epsInterval,eps_def,minptsInterval)
    num_points = size(M,1);
    Distance = zeros(num_points);
    
    for i=1:num_points
        for j=i:num_points
            Distance(i,j) = norm( M(i,:)-M(j,:) );
            Distance(j,i) = Distance(i,j);
        end
    end
    avrg_dist = sum(sum(Distance)/size(Distance,1))/size(Distance,1);
%     eps = (0.19:0.01:0.23)*avrg_dist; 
    eps = (epsInterval(1):eps_def:epsInterval(2))*avrg_dist;
    num_eps = size(eps,2);
    minpts = minptsInterval(1):minptsInterval(2);
    num_minpts = size(minpts,2);

    % % % % %DEBUG!!! COMENT LATER!! TO TEST COMPONENT DETECTION
    % % % % i=2;
    % % % % j=1;
    % % % % [idx,~] = dbscan(M,eps(i),minpts(j));
    % % % % num_clusters = max(idx);
    % % % % for k=1:num_clusters
    % % % %     scatter3(M(idx==k,1),M(idx==k,2),M(idx==k,3),'.')
    % % % %     hold on
    % % % % end
    % % % % scatter3(M(idx==-1,1),M(idx==-1,2),M(idx==-1,3),'x')


    WLC_min = inf;
%     WLC_max = 0;
    C_fin = {};
    eps_fin = 0;
    minpts_fin = 0;
    num_clust_fin = 0;
    idx_fin = [];
    for i=1:num_eps
%         disp(['Current Epsilon:' int2str(i) '/' int2str(num_eps) ])
        for j=1:num_minpts
            [idx,~] = dbscan(M,eps(i),minpts(j));
            num_clusters = max(idx);
            C = {};
            for k=1:num_clusters
                C{k} = M(find(idx==k),:);
            end
            WLC = WeakestLinkCluster(C);
%             disp([['WLC= ' num2str(WLC) ', eps=' int2str(i) ', minpts=' int2str(minpts(j)) ', ' int2str(num_clusters) ' clusters']])
            if WLC < WLC_min
%             if WLC > WLC_max
                WLC_min = WLC;
%                 WLC_max = WLC;
%                 disp(['NUEVO WLC= ' num2str(WLC) ', eps=' int2str(i) ', minpts=' int2str(minpts(j)) ', ' int2str(num_clusters) ' clusters'])
                eps_fin = eps(i);
                C_fin = C;
                minpts_fin = minpts(j);
                num_clust_fin = num_clusters;
                idx_fin = idx;
            end
        end
    end
end
%-------------------------%
function WLC = WeakestLinkCluster(C)

    num_clust = size(C,2);
%     inter_clust_WLP = zeros(1,num_clust);

    %---This part computes the maximum link intra cluster-----%
    C_sizes = zeros(1,num_clust);
    max_clust = zeros(1,num_clust); %MAXIMUM LINK INTRA CLUSTER
    for i=1:num_clust
        C_sizes(i) = size(C{i},1);
    end
    for i=1:num_clust %compute value over all clusters
%         if C
        for j=1:C_sizes(i)-1 %find maximum dist in cluster
            dist = norm(C{i}(j,:)-C{i}(j+1,:));
            if dist > max_clust(i)
                max_clust(i) = dist;
            end
        end
    end
    %---------------------------------------------------------%

    %----COMPUTING INTRA CLUSTER WLP-----------%
    intra_cluster_WLP = 0;
    for i=1:num_clust-1
        for j=1:C_sizes(i)-1
            for k=j+1:C_sizes(i)
                temp = WeakestLinkPoints(max_clust,C{i}(j,:),C{i}(k,:),C);
                if temp > intra_cluster_WLP
                    intra_cluster_WLP = temp;
                end
            end
        end
    end
    %------------------------------------------%

    %----SHORTEST BETWEEN CLUSTER DISTANCE------%
    inter_clust_WLP = inf;
    for i=1:num_clust-1
        for j=i+1:num_clust
            temp = min_dist_2_clust(C{i},C{j});
            if temp < inter_clust_WLP
                inter_clust_WLP = temp;
            end
        end
    end
    %------------------------------------------%

    WLC = intra_cluster_WLP/inter_clust_WLP;


end
%-------------------------%
function WLP = WeakestLinkPoints(max_clust,x,y,C)
    num_clust = size(C,2);
    WLP_vec = zeros(1,num_clust);
    dist_vec = zeros(1,3);
    for i=1:num_clust
        dist_vec(1) = norm(C{i}(1,:)-x);
        dist_vec(2) = norm(C{i}(end,:)-y);
        dist_vec(3) = max_clust(i);
        WLP_vec(i) = max(dist_vec);
    end
    WLP = min(WLP_vec);
end
%-------------------------%
function min = min_dist_2_clust(X,Y)
    min = inf;
    X_size = size(X,1);
    Y_size = size(Y,1);
    for i=1:X_size
        for j=1:Y_size
            temp = norm(X(i,:)-Y(j,:));
            if temp < min
                min = temp;
            end
        end
    end
end
%-------------------------%

%-------------------------------------------------------------------------%
