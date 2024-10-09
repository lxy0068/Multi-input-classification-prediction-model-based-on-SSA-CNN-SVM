function Positions=initialization(SearchAgents_no, dim, ub, lb)

%%  �߽���Ŀ
Boundary_no= size(ub, 2);

%%  ������Ŀ����1
if Boundary_no == 1
    Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
end

%% ���ÿ�������в�ͬ�����½�
if Boundary_no > 1
    for i = 1 : dim
        ub_i = ub(i);
        lb_i = lb(i);
        Positions(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
    end
end