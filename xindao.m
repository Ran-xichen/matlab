%% AWGN 信道建模
clc, close all, clear all;

% 定义高斯分布参数
mu = 2;               % 均值为2
sigma_values = [1, 2, 3, 4]; % 不同方差值
num_samples = 10000;  % 生成的样本数量

% 绘制不同方差下的高斯分布曲线
figure;
hold on;
colors = lines(length(sigma_values)); % 使用不同颜色
legend_labels = cell(1, length(sigma_values)); % 图例标签

for i = 1:length(sigma_values)
    sigma = sigma_values(i);
    
    % 生成高斯分布随机变量
    samples = mu + sigma * randn(num_samples, 1);
    
    % 计算样本均值和方差
    sample_mean = mean(samples);
    sample_var = var(samples);
    
    % 显示结果
    fprintf('方差: %.4f\n', sigma^2);
    fprintf('样本均值: %.4f\n', sample_mean);
    fprintf('样本方差: %.4f\n\n', sample_var);
    
    % 绘制理论高斯分布曲线
    x = linspace(mu - 4*sigma, mu + 4*sigma, 1000);
    pdf_theoretical = normpdf(x, mu, sigma);
    plot(x, pdf_theoretical, 'Color', colors(i, :), 'LineWidth', 2);
    
    % 图例标签
    legend_labels{i} = sprintf('方差 = %.1f', sigma^2);
end

% 图形标注
title('不同方差下的高斯分布');
xlabel('值');
ylabel('概率密度');
legend(legend_labels, 'Location', 'best');
grid on;
hold off;