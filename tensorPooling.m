clc
clear all
close all

pn = 'testingDataset\original\';
pn2 = 'testingDataset\test_images\';

ext_img = [pn '*.png'];
a = dir(ext_img);
nfile = length(a);
proposals = {};
obj = [];

for i=1:nfile
    fn = a(i).name; 
    img = imread([pn fn]);

    if ismatrix(img) == false
        img = rgb2gray(img);
    end
    
    img = imadjust(img);
    
    N = 4;
    n = 3;
    i1 = img;
    p = {};
    s = {};
    i2 = zeros(size(i1));
%     [s1,s2,s3]=stOld(i1,1,10);
%     imagesc(s1)
    tic   
    for j = 1:n
        i1 = impyramid(i1,'reduce');
        i3 = i1;
        tensors = structureTensor(i1,N);
        [c1,c2] = getCoherentOne(tensors);
        if ~isempty(c2)
            s1 = c1 + c2;
        end
%         [s1,s2,s3]=stOld(i1,1,10);
%         s1 = s1 + s2 + s3;
%         imagesc(s1)
        s1 = s1(10:end-10,10:end-10);
        s{j} = s1;
        figure,imagesc(s1)
        if j > 1
            s4 = imresize(s{j-1},size(i1),'bilinear');
            i1 = double(i1) .* s4;
        else
            i1 = imresize(i1,size(s1),'bilinear');
            i1 = double(i1) .* s1;
        end
        
        p{j} = i1;
        i1 = imresize(i1,size(i2),'bilinear');
        i2 = i2 + double(i1);
        i1 = i3;
        close all
    end
    toc
    figure,imagesc(i2)
    c = getframe
    c = c.cdata;

	c = imresize(c,[576 768],'bilinear');
%     fn=replace(fn,'.jpg','.png');

    imwrite(c,[pn2 fn],'PNG');
end

function [cTensor,cTensor2] = getCoherentOne(tensors)
cTensor = [];
cTensor2 = [];
[r,c] = size(tensors);
m = -100000000000000000;
index1 = -1;
index2 = -1;
for i = 1:r
    for j = 1:c
        if isempty(tensors{i,j}) 
            continue;
        end
        
        t = tensors{i,j};
        [u s v] = svd(t);
        if m < max(max(s))
            cTensor2 = cTensor;
            cTensor = t;
            m = max(max(s));
        end
    end
end

% figure,imagesc(cTensor)
end


function [Sxx, Sxy, Syy] = stOld(I,si,so)
I = double(I);
[m n] = size(I);
 
Sxx = NaN(m,n);
Sxy = NaN(m,n);
Syy = NaN(m,n);
 
x  = -2*si:2*si;
g  = exp(-0.5*(x/si).^2);
g  = g/sum(g);
gd = -x.*g/(si^2); 
 
Ix = conv2( conv2(I,gd,'same'),g','same' );
Iy = conv2( conv2(I,gd','same'),g,'same' );
 
Ixx = Ix.^2;
Ixy = Ix.*Iy;
Iyy = Iy.^2;
 
x  = -2*so:2*so;
g  = exp(-0.5*(x/so).^2);
Sxx = conv2( conv2(Ixx,g,'same'),g','same' ); 
Sxy = conv2( conv2(Ixy,g,'same'),g','same' );
Syy = conv2( conv2(Iyy,g,'same'),g','same' );

end
%%
function [tensors] = structureTensor(I,N)

I = double(I);
[m n] = size(I);
si = 1;
so = 1;
tensors = {};
Sxx = NaN(m,n);
Sxy = NaN(m,n);
Syy = NaN(m,n);
 
x  = -2*si:2*si;
g  = exp(-0.5*(x/si).^2);
g  = g/sum(g);
gd = -x.*g/(si^2); 

a = zeros(5,5);
a(3,:) = gd;
b = zeros(5,5);
b(:,3) = g;
an = a;
index = 1;

gradients = {};

index = 1;
for i = 0:N-1
    angle = (2*180*i)/N;
    a = imrotate(an,angle,'bilinear','crop');
    Ig = conv2( conv2(I,a,'same'),b','same' );
    gradients{index} = Ig;
    index = index + 1;
end

nGradients = length(gradients);

for i = 1:nGradients
    I1 = gradients{i};
    for j = 1:nGradients
        I2 = gradients{j};
        Ixy = I1.*I2;
        Sxy = imdiffusefilt(Ixy);
        x  = -2*so:2*so;
        g  = exp(-0.5*(x/so).^2);
%         Sxy = conv2( conv2(Ixy,g,'same'),g','same' );
        tensors{i,j} = Sxy;
%         imagesc(Sxy)
    end
end
end