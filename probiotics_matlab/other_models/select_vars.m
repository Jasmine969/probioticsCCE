% input a matrix, each of whose row represents a group of samples and whose
% column represents a variable, e.g. X:
% ta  va   ws  vd
% 70  0.75 0.1 1
% 90  0.75 0.2 1
% 110 1    0.1 2
% a cell storing the var-names, and input a number between 1 and 2^n,
% where n is the number of the
% variables. The programme will transform n to binary to select the
% variables to be fitted.
function X_selected = select_vars(X, names, n)
    % select var
    select_ind = logical(bitget(n, size(X,2):-1:1));
    % select and transpose
    names(:, select_ind)
    X_selected = X(:, select_ind);
end