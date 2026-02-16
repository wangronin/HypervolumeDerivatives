%%%-----------------------V1.3.5 09/07/2024-----------------------------%%%
function eta = shift_vect(Py,show_plots)
    
    k = size(Py,2);
    
    if size(Py,1) <k
        eta = ones(1,k);
    else
        [~,I] = max(Py);
        if length(unique(I)) ~= k
            %SOLUCION CLUSTER
%             [~,~,~,~,I] = kmedoids(Py,k,'Options',statset('MaxIter',10));%MODIFICACION
            %SOLUCION RANDOM
%             for i=1:k
%                 index = find(Py(:,i)==Py(I(i),i));
%                 I(i) = max(index); %max
%         %         I(i) = index(randi(length(index))); %rand
%             end
            %SOLUCION MIN
            [~,I] = min(Py);
            if length(unique(I)) ~= k
                %SOLUCION CLUSTER
                [~,~,~,~,I] = kmedoids(Py,k,'Options',statset('MaxIter',10));%MODIFICACION
            end
        end
        
        if show_plots
            figure
            scatter3(Py(:,1),Py(:,2),Py(:,3),'.')
            hold on
            scatter3(Py(I,1),Py(I,2),Py(I,3),'filled')
            title('ENDPOINTS DETECTED')
        end

        %         [~,~,~,~,I] = kmedoids(Py,k);%MODIFICACION
        M = zeros(k,k-1);
        for i=1:k-1
            M(:,i) = Py(I(i+1),:) - Py(I(1),:);
        end
        
        [Q,R] = qr(M);
        
        qk = Q(:,end);
        % qk = Q(1,:);
        
        eta = -1*sign(qk(1))*qk;
        % eta = qk;
        
        eta = eta/norm(eta);
    end
end