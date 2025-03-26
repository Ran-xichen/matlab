%% AWGN �ŵ���ģ
clc, close all, clear all;

% �����˹�ֲ�����
mu = 2;               % ��ֵΪ2
sigma_values = [1, 2, 3, 4]; % ��ͬ����ֵ
num_samples = 10000;  % ���ɵ���������

% ���Ʋ�ͬ�����µĸ�˹�ֲ�����
figure;
hold on;
colors = lines(length(sigma_values)); % ʹ�ò�ͬ��ɫ
legend_labels = cell(1, length(sigma_values)); % ͼ����ǩ

for i = 1:length(sigma_values)
    sigma = sigma_values(i);
    
    % ���ɸ�˹�ֲ��������
    samples = mu + sigma * randn(num_samples, 1);
    
    % ����������ֵ�ͷ���
    sample_mean = mean(samples);
    sample_var = var(samples);
    
    % ��ʾ���
    fprintf('����: %.4f\n', sigma^2);
    fprintf('������ֵ: %.4f\n', sample_mean);
    fprintf('��������: %.4f\n\n', sample_var);
    
    % �������۸�˹�ֲ�����
    x = linspace(mu - 4*sigma, mu + 4*sigma, 1000);
    pdf_theoretical = normpdf(x, mu, sigma);
    plot(x, pdf_theoretical, 'Color', colors(i, :), 'LineWidth', 2);
    
    % ͼ����ǩ
    legend_labels{i} = sprintf('���� = %.1f', sigma^2);
end

% ͼ�α�ע
title('��ͬ�����µĸ�˹�ֲ�');
xlabel('ֵ');
ylabel('�����ܶ�');
legend(legend_labels, 'Location', 'best');
grid on;
hold off;