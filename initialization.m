function Positions=initialization(SearchAgents_no, dim, ub, lb)

%%  边界数目
Boundary_no= size(ub, 2);

%%  变量数目等于1
if Boundary_no == 1
    Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
end

%% 如果每个变量有不同的上下界
if Boundary_no > 1
    for i = 1 : dim
        ub_i = ub(i);
        lb_i = lb(i);
        Positions(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
    end
end