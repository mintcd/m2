function plot_errors(all_errors)
    nranks = size(all_errors, 1);  % Number of ranks
    
    figure;
    hold on;  % Keep the plot active to overlay multiple lines
    
    for r = 1:nranks
        iters = length(all_errors{r});  % Length of error vector for rank r
        semilogy(1:iters, all_errors{r}, 'DisplayName', ['rank ' num2str(r)]);
    end
    
    xlabel('iteration');
    ylabel('error');
    title('Convergence of CP Decomposition for Different Ranks (Log Scale)');
    legend show;  % Show the legend with rank labels
    hold off;  % Release the plot
end
