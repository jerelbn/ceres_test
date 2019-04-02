% Plot data from LC-VI-MHE test
clear, close all
set(0,'DefaultFigureWindowStyle','docked')

% load the data
truth_file = fopen('../logs/true_state.bin', 'r');
truth = reshape(fread(fopen('../logs/true_state.bin', 'r'), 'double'), 16, []);

% plot truth
titles = {'Position', 'Velocity', 'Acceleration', 'Euler Angles', 'Angular Velocity'};
for i = 1:5
    figure(i), clf
    set(gcf, 'color', 'w')
    set(gcf, 'name', titles{i}, 'NumberTitle', 'off')
    for j = 1:3
        subplot(3,1,j)
        plot(truth(1,:), truth(3*(i-2/3)+j,:), 'b-', 'LineWidth', 1.5)
        grid on
        if j == 1
            title(titles{i})
        end
    end
end
