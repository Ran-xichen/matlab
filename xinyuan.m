%%
% AI指令:请用matlab 仿真通信系统，要求输出误比特率和信噪比之间的仿真曲线图，要求信噪比为-6：8
clc,close all,clear all;
% 参数设置
numBits = 1e6; % 传输的比特数
snrRange = -6:8; % 信噪比范围
ber = zeros(size(snrRange)); % 初始化误比特率数组

% 主循环：遍历不同的信噪比
for i = 1:length(snrRange)
    snr = snrRange(i); % 当前信噪比
    % 生成随机比特序列
    txBits = randi([0 1], 1, numBits);
    % BPSK调制：0 -> -1, 1 -> +1
    txSignal = 2 * txBits - 1;
    % 添加高斯白噪声
    rxSignal = awgn(txSignal, snr, 'measured');
    % BPSK解调：接收信号 > 0 -> 1, 接收信号 < 0 -> 0
    rxBits = rxSignal > 0;
    % 计算误比特率
    ber(i) = sum(rxBits ~= txBits) / numBits;
end

% 绘制误比特率与信噪比的关系曲线
semilogy(snrRange, ber, 'b-o', 'LineWidth', 2);
grid on;
xlabel('信噪比 (SNR) [dB]');
ylabel('误比特率 (BER)');
title('BPSK调制在AWGN信道中的误比特率性能');

%% 瑞利分布信源建模与验证
clc; clear; close all;

% 参数设置
N = 1e5;        % 样本数
sigma = 1;      % 高斯分量标准差

% 生成瑞利分布信源
W1 = sigma * randn(1, N);
W2 = sigma * randn(1, N);
X = sqrt(W1.^2 + W2.^2);

% 绘制PDF对比
[counts, edges] = histcounts(X, 50, 'Normalization', 'pdf');
bin_centers = (edges(1:end-1) + edges(2:end))/2;
x_theory = linspace(0, max(X), 1000);
pdf_theory = (x_theory / sigma^2) .* exp(-x_theory.^2 / (2*sigma^2));

figure;
bar(bin_centers, counts); hold on;
plot(x_theory, pdf_theory, 'r', 'LineWidth', 2);
xlabel('幅度'); ylabel('概率密度');
title(['瑞利分布（σ=' num2str(sigma) '）仿真验证']);
legend('仿真直方图', '理论曲线');
grid on;

%% 莱斯分布信源建模与验证
clc; clear; close all;

% 参数设置
N = 1e5;        % 样本数
K_dB = 10;      % 莱斯因子K（dB）
sigma = 1;      % 瑞利分量标准差

% 计算莱斯参数
K = 10^(K_dB/10);
beta = sqrt(2*K*sigma^2);  % 直射路径幅度?:ml-citation{ref="3,5" data="citationList"}

% 生成莱斯分布信号
W1 = sigma * randn(1, N);
W2 = sigma * randn(1, N);
X = sqrt((beta + W1).^2 + W2.^2); 

% 绘制PDF对比
[counts, edges] = histcounts(X, 50, 'Normalization', 'pdf');
bin_centers = (edges(1:end-1) + edges(2:end))/2;

% 理论PDF计算
x_theory = linspace(0, max(X), 1000);
A = beta / sigma;
s = sigma;
pdf_theory = (x_theory / s^2) .* exp(-(x_theory.^2 + A^2*s^2)/(2*s^2)) .* besseli(0, (x_theory*A)/s^2);

% 绘图
figure;
bar(bin_centers, counts); hold on;
plot(x_theory, pdf_theory, 'r', 'LineWidth', 2);
xlabel('幅度'); ylabel('概率密度');
title(['莱斯分布（K=' num2str(K_dB) 'dB）仿真验证']);
legend('仿真直方图', '理论曲线');
grid on;

%% 泊松分布信源建模与验证
clc; clear; close all;
% 参数设置
N = 1e5;        % 样本数量
lambda = 5;     % 平均发生率

% 生成泊松分布信源
X = poissrnd(lambda, 1, N);    % 调用内置函数生成泊松随机变量?:ml-citation{ref="4" data="citationList"}

% 绘制PMF对比
k_values = 0:max(X);
pmf_sim = histcounts(X, [k_values-0.5, max(k_values)+0.5], 'Normalization', 'probability');
pmf_theory = poisspdf(k_values, lambda);

figure;
stem(k_values, pmf_sim, 'bo', 'MarkerSize', 5, 'LineWidth', 1.5); hold on;
plot(k_values, pmf_theory, 'r*', 'LineWidth', 1.5);
xlabel('k'); ylabel('概率');
title(['泊松分布（λ=' num2str(lambda) '）PMF对比']);
legend('仿真数据', '理论曲线');
grid on;

%% 单符号离散无记忆信源
P0=0.3;
P1=0.7;
N=1000;
x=randsrc(1,N,[0 1;P0 P1]);
N0=length(find(x==0));
P0x=N0/N;
P1x=1-P0x;
H1=-log2(P0)-log2(P1);
H2=-log2(P0x)-log2(P1x);
fprintf('x=\t');
for i = 1:length(x)
    fprintf('%d ', x(i));
end
fprintf('\n'); % 换行
fprintf('理论P0x= %.4f\n', P0);
fprintf('理论P1x= %.4f\n', P1);
fprintf('实际P0x= %.4f\n', P0x);
fprintf('实际P1x= %.4f\n', P1x);
fprintf('理论熵H= %.4f\n', H1);
fprintf('实际熵H= %.4f\n', H2);







































