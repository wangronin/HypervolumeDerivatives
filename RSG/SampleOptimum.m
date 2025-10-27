function Py = SampleOptimum(problem_name,ref_size) 
    switch problem_name
        case '1_over_x'
            x = linspace(0.1,3,ref_size);
            f1 = 1 - 1./x;
            f2 = 1./x;
            Py = [f1',f2'];
        case 'CONV3'
            Py = [];
        case 'CONV3MM'
            Py = [];
            delta1 = 3;
            delta2 = 6;
            evil = [6,6,6];
            num_obj = 3;
            num_var = 3;
            a = {};
            a{1} = -1*ones(1,num_var);
            a{2} = ones(1,num_var);
            a{3} = (-1).^(1:num_var);
            Ftemp = zeros(1,num_obj);
            y = zeros(1,num_obj);
            if x(1) <= -4
                for i=1:num_obj
                    Ftemp(i) = norm(x-a{i}+evil)^2   +(delta1);
                end
            end
            if x(1) < 2 && x(1)>-4
               for i=1:num_obj
                   Ftemp(i) = norm(x-a{i})^2;
               end
            end
            if x(1) >= 2
               for i=1:num_obj
                   Ftemp(i) = norm(x-a{i}-evil)^2   +(delta2);
               end
            end
            
            for i=1:num_obj
               y(i) = (1-alpha)*Ftemp(i)+(alpha/num_obj)*sum(Ftemp);
            end
             
        case 'CONV3_2'
            Py = [];
        case 'CONV4'
            num_obj = 4;
            num_var = 4;
            a = {};
            for i=1:num_var
                a{i} = zeros(1,num_var);
                a{i}(i) = 1;
            end
            Ftemp = zeros(1,num_obj);
            y = zeros(1,num_obj);

            for i=1:num_obj
                Ftemp(i) = norm(x-a{i})^2;
            end
             
            Py = [];
        case 'CONV4_2F'
            Py = [];
        otherwise
            Py = [];
    end
end