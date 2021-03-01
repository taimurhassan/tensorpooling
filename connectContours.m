function arr = connectContours(arr,rMax)
% Checks each pixel of a logical array of contours to see if it is connected to other contour's 
% pixels.
% If not connected, it attempts to draw connections to pixels of the contours via Bresenham's 
% straight line algorithm.

% load edgeVecs % pre-calculated list of perimeter coordinates to save processing time
% Generated using this code:
N = rMax;%prod(size(arr));
for r = 1:N
	s = 2*r+1;
	edgeVecs{1,r} = [1:s repmat(s,[1 s-1]) s-1:-1:1 ones([1 s-2])];
	edgeVecs{2,r} = [ones([1 s-1]) 1:s repmat(s,[1 s-1]) s-1:-1:2];
end

d = size(arr);
for x = 1:d(2)
for y = 1:d(1)
	if arr(y,x)
		connected = 0; r=0;
		while connected < 2 && r <= rMax
			r=r+1;
			xVec = edgeVecs{1,r}+x-r-1;	yVec = edgeVecs{2,r}+y-r-1;  % Indicies of pixels r-away from (y,x)
			xVec(xVec<1)=1;				yVec(yVec<1)=1;				 % Avoid index violations
			xVec(xVec>d(2))=d(2);		yVec(yVec>d(1))=d(1);
			for ndx = 1:numel(xVec)
				cX = xVec(ndx); cY = yVec(ndx);
				if arr(cY,cX)							    % A kindred pixel has been found
					sX = 1; if x-cX<0 sX=-1;end				% Step direction for indexing
					sY = 1; if y-cY<0 sY=-1;end
					if sum(sum(arr(cY:sY:y,cX:sX:x))) == 2  % No intermediate pixels
						connected = connected+1;
						if r>1
						[xPts,yPts] = bresenham(cX,cY,x,y); % Draw line to connect hem
						arr(sub2ind(d,yPts,xPts)) = true;
						end
					end
				end
			end
		end
	end
end
end
end

function [x,y]=bresenham(x1,y1,x2,y2)
% Line to pixel approximation algorithm (Bresenham's line algorithm)
% Credit: Aaron WetzlerAll (2010)
dx=abs(x2-x1);dy=abs(y2-y1);
steep=abs(dy)>abs(dx);
if steep t=dx;dx=dy;dy=t; end

if dy==0 
    q=zeros([dx+1,1]);
else
    q=[0;diff(mod((floor(dx/2):-dy:-dy*dx+floor(dx/2))',dx))>=0];
end

if steep
    if y1<=y2 y=(y1:y2)'; else y=(y1:-1:y2)'; end
    if x1<=x2 x=x1+cumsum(q);else x=x1-cumsum(q); end
else
    if x1<=x2 x=(x1:x2)'; else x=(x1:-1:x2)'; end
    if y1<=y2 y=y1+cumsum(q);else y=y1-cumsum(q); end
end
end
