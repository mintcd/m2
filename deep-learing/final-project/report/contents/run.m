img_com = image_compression('./bigben.jpg');

figure = tiledlayout(2,4);
nexttile;
imshow(A);
title('Original');
for k = [1 5 10 25 50 100 200]
    [Ak,inf_k] = img_com.approx(k);
    nexttile
    imshow(Ak)
    title(sprintf("Rank: %d, explained var: %.2f%%", k, inf_k*100))
end