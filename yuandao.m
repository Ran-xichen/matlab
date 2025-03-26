% AWGN 信道建模与模型验证
clc;close all;clear all;

% 参数设置
num_samples = 10000;       % 信号样本数量
snr_dB = 10;               % 信噪比 (dB)
signal_power = 1;          % 信号功率 (假设为单位功率)

% 生成发送信号 (随机信号)
tx_signal = sqrt(signal_power) * (2 * randi([0, 1], num_samples, 1) - 1); % BPSK 信号

% 计算噪声功率
noise_power = signal_power / (10^(snr_dB / 10)); % 根据 SNR 计算噪声功率

% 生成 AWGN 噪声
noise = sqrt(noise_power) * randn(num_samples, 1); % 高斯噪声

% 通过 AWGN 信道
rx_signal = tx_signal + noise; % 接收信号 = 发送信号 + 噪声

% 模型验证
% 1. 验证噪声分布
figure;
subplot(2, 1, 1);
histogram(noise, 50, 'Normalization', 'pdf', 'FaceColor', [0.8, 0.8, 1]);
hold on;
x = linspace(min(noise), max(noise), 1000);
pdf_theoretical = normpdf(x, 0, sqrt(noise_power)); % 理论高斯分布
plot(x, pdf_theoretical, 'r-', 'LineWidth', 2);
title('噪声分布验证');
xlabel('噪声值');
ylabel('概率密度');
legend('噪声直方图', '理论PDF');
grid on;

% 2. 计算噪声的均值和方差
noise_mean = mean(noise);
noise_var = var(noise);
fprintf('噪声均值: %.4f (理论值: 0)\n', noise_mean);
fprintf('噪声方差: %.4f (理论值: %.4f)\n', noise_var, noise_power);

% 3. 计算信号的误差 (发送信号与接收信号的差异)
signal_error = rx_signal - tx_signal;
fprintf('信号误差的均值: %.4f\n', mean(signal_error));
fprintf('信号误差的方差: %.4f\n', var(signal_error));

% 4. 绘制发送信号与接收信号的对比
subplot(2, 1, 2);
plot(1:100, tx_signal(1:100), 'bo-', 'LineWidth', 1.5, 'DisplayName', '发送信号');
hold on;
plot(1:100, rx_signal(1:100), 'rx-', 'LineWidth', 1.5, 'DisplayName', '接收信号');
title('发送信号与接收信号对比 (前 100 个样本)');
xlabel('样本索引');
ylabel('信号值');
legend;
grid on;
hold off;

% 性能分析
% 计算误码率 (BER)
decoded_signal = rx_signal > 0; % BPSK 解调
bit_errors = sum(decoded_signal ~= (tx_signal > 0)); % 误码数
ber = bit_errors / num_samples; % 误码率
fprintf('误码率 (BER): %.4f\n', ber);