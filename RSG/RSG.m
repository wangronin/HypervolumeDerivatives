%%%-----------------------V1.3.4 01/07/2024-----------------------------%%%
% function [Z_means,Z_medoids] = RSG(Py,Nf,N,clean_method,threshold,name,epsInterval,eps_def,minptsInterval,show_plots,save_plots)
function [Z_means,Z_medoids] = RSG()

Nf = 10000;
N = 100;
clean_method = 'off';
threshold = 'off';
name = 'MMD_boostrap';
epsInterval = [10.2,10.2];
eps_def = 0.02;
minptsInterval = [2,2];
show_plots = false;
save_plots = false;
Py = csvread("MMD_boostrap.csv");

if isempty(clean_method)
    clean_method = 'long'; % Default value for clean_method
end
if isempty(threshold)
    threshold = 3; % Default value for threshold
end

k = size(Py,2);
cluster_iters = 500;

%----Interpolate--%
Iy = interpolation( Py, Nf, clean_method, threshold, name, epsInterval, eps_def, minptsInterval, show_plots, save_plots );
% figure
% if size(Iy,2) == 2
%     scatter(Iy(:,1),Iy(:,2),'.')
% else
%     scatter3(Iy(:,1),Iy(:,2),Iy(:,3),'.')
%     zlabel('f_3')
% end
if show_plots
    figSettings
    xlabel('f_1')
    ylabel('f_2')
    title('Filling')
    if save_plots
        saveas(gcf, [ pwd '/PLOTS/' name '_Filling.fig' ] );
        saveas(gcf, [ pwd '/PLOTS/' name '_Filling.pdf' ] );
        saveas(gcf, [ pwd '/PLOTS/' name '_Filling.png' ] );
    end
end
%-----Trimming----%
% [~,Z_means] = kmeans(Iy,N,'Options',statset('MaxIter',cluster_iters));
% [~,Z_medoids,~,~,~] = kmedoids(Iy,N,'Options',statset('MaxIter',cluster_iters));

k = size(Iy,2);
[~,I] = max(Iy);
if length(unique(I)) ~= k
    %SOLUCION MIN
    [~,I] = min(Iy);
end
[~,Z_means] = kmeans(Iy,N-k,'Options',statset('MaxIter',cluster_iters));
[~ ,Z_medoids,~, ~, ~] = kmedoids(Iy,N-k,'Options',statset('MaxIter',cluster_iters));
Z_means(end+1:end+k,:) = Iy(I,1:k);
Z_medoids(end+1:end+k,:) = Iy(I,1:k);

if show_plots
    figure
%     close all
    if k==2
        scatter(Z_means(:,1),Z_means(:,2),'d','filled')
        hold on
        scatter(Py(:,1),Py(:,2),'.')
    else
        scatter3(Z_means(:,1),Z_means(:,2),Z_means(:,3),'filled')
        hold on
        scatter3(Py(:,1),Py(:,2),Py(:,3),'.')
        zlabel('f_3')
    end
    legend('Z','Py')
%     figSettings
    title('Final Result (Z) vs Start (Py)')
    xlabel('f_1')
    ylabel('f_2')
    if save_plots
        saveas(gcf, [ pwd '/PLOTS/' name '_RSG.fig' ] );
        saveas(gcf, [ pwd '/PLOTS/' name '_RSG.pdf' ] );
        saveas(gcf, [ pwd '/PLOTS/' name '_RSG.png' ] );
    end
end

end
