% Image Compression Class
classdef image_compression
   properties
      U, S, V;
   end
   methods        
       function obj = image_compression(img_path)
           img = imread(img_path);
           img = rgb2gray(img);

           [obj.U,obj.S,obj.V] = svd(double(img));   
       end

       function [approximated_img, inf] = approx(obj, rank)
           U_k = obj.U(:,1:rank);
           S_k = obj.S(1:rank,1:rank);
           V_k = obj.V(:,1:rank);

           inf = sum(diag(S_k).^2)/sum(diag(obj.S).^2);

           approximated_img = uint8(U_k*S_k*V_k');
       end
    end
end


