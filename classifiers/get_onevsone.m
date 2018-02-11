%pairwise coupling
%Code by Wu, et.al 2004

%input: matrix KxK

function [pest]=get_onevsone(r)

%%% second approach
k=size(r,1);
B = -r.*(r');
for i=1:k, 
  B(i,i) =  sum(r(:,i).*r(:,i));
end
p = [B ones(k,1); ones(1,k) 0] \ [zeros(k,1); 1];
pest = p(1:k,1);

%%% first approach
% k=size(r,1);
% C = r + diag(sum(r,2)-(k-1));
% p =[C(1:k-1,:); ones(1,k)]\[zeros(k-1,1); 1];


%%% iterative approach
%%% by Hastie and Tibshirani, 1998
% k = size(r,1);
% 
% pest = sum(r,2)*2/k/(k-1);
% 
% maxiter=1000;
% pest = pest / sum(pest);
% mu=zeros(k,k);
% for i =1:k
%   mu(i,i+1:k) = pest(i) ./ (pest(i)+pest(i+1:k)') ;
%   mu(i+1:k,i) = 1 - mu(i,i+1:k)';
% end
% 
% for t=1:maxiter,
%   for i = 1:k,
%     alpha =  sum(r(i,:)) / sum(mu(i,:)) ;      
%     pest(i) = pest(i)*alpha;
%     noti=[1:i-1,i+1:k];
%     mu(i,noti) = alpha*mu(i,noti) ./ (alpha*mu(i,noti)+ mu(noti,i)'); 
%     mu(noti,i) = 1 - mu(i,noti)'; 
%   end
%   pest = pest ./ sum(pest);
%   if (max(abs(sum(r,2)./sum(mu,2)-1)) < 0.001)
%     return;
%   end
% end
% fprintf(1, 'max iteration\n');

