function [er, bad] = cnntest(net, x, y)
    %  feedforward
    net = cnnff(net, x);
    net.o
    b = round(net.o);
    bad = find((b - y) ~= 0);
    er = size(bad,2)/size(y,2);   
    er
end
