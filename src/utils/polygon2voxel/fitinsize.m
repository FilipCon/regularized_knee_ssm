function y = fitinsize(x, mini, maxi)

y = x;
if y<mini, 
    y=mini; 
elseif y>maxi, 
    y=maxi;
end
