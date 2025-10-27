%%%-----------------------V1.3.4 01/07/2024-----------------------------%%%
function [ DT_clean ] = rmtriangle( DT, Py, fun_param, threshold, comp_name, show_plots, save_plots )
    
    num_obj = size(Py,2);
    num_bins = 40;

	TA = [];
	for i = 1:size(DT,1)
		TA(i) = fun_param( Py(DT(i,:),1:end) );
	end
    
    if show_plots
        figure
	    hist = histogram( TA, num_bins );
	    title( 'Histogram');
        figSettings
        if save_plots
            saveas(gcf, [ pwd '/PLOTS/' comp_name '_histogram.fig' ] );
            saveas(gcf, [ pwd '/PLOTS/' comp_name '_histogram.pdf' ] );
            saveas(gcf, [ pwd '/PLOTS/' comp_name '_histogram.png' ] );
        end
    end

    if strcmp(threshold,'zero')
        first_zero = find(hist.Values == 0);
        if isempty(first_zero)
            T = inf;
        else
            T = hist.BinEdges(first_zero(1));
        end
    elseif strcmp(threshold,'decreasing')
        first_decreasing = find(diff(hist.Values)<0);
        if isempty(first_decreasing)
            T = inf;
        else
        T = hist.BinEdges(first_decreasing(1)+1);
        end
    else
%         T = mean(TA)*threshold;
        T = threshold;
    end

	
	TA2 = [];
	DT_clean = [];
	for i = 1:size(TA,2)
		% Accept
		if TA(i) <= T
			TA2(end+1) = TA(i);
			DT_clean(end+1,:) = DT(i,:);
		end
	end

    if show_plots
        figure
	    hist = histogram( TA2, num_bins );
	    title('Histogram Cleaned');
        figSettings
        if save_plots
            saveas(gcf, [ pwd '/PLOTS/' comp_name '_histogram_clean.fig' ] );
            saveas(gcf, [ pwd '/PLOTS/' comp_name '_histogram_clean.pdf' ] );
            saveas(gcf, [ pwd '/PLOTS/' comp_name '_histogram_clean.png' ] );
        end
    end

end