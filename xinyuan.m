%%
% AIָ��:����matlab ����ͨ��ϵͳ��Ҫ�����������ʺ������֮��ķ�������ͼ��Ҫ�������Ϊ-6��8
clc,close all,clear all;
% ��������
numBits = 1e6; % ����ı�����
snrRange = -6:8; % ����ȷ�Χ
ber = zeros(size(snrRange)); % ��ʼ�������������

% ��ѭ����������ͬ�������
for i = 1:length(snrRange)
    snr = snrRange(i); % ��ǰ�����
    % ���������������
    txBits = randi([0 1], 1, numBits);
    % BPSK���ƣ�0 -> -1, 1 -> +1
    txSignal = 2 * txBits - 1;
    % ��Ӹ�˹������
    rxSignal = awgn(txSignal, snr, 'measured');
    % BPSK����������ź� > 0 -> 1, �����ź� < 0 -> 0
    rxBits = rxSignal > 0;
    % �����������
    ber(i) = sum(rxBits ~= txBits) / numBits;
end

% �����������������ȵĹ�ϵ����
semilogy(snrRange, ber, 'b-o', 'LineWidth', 2);
grid on;
xlabel('����� (SNR) [dB]');
ylabel('������� (BER)');
title('BPSK������AWGN�ŵ��е������������');

%% �����ֲ���Դ��ģ����֤
clc; clear; close all;

% ��������
N = 1e5;        % ������
sigma = 1;      % ��˹������׼��

% ���������ֲ���Դ
W1 = sigma * randn(1, N);
W2 = sigma * randn(1, N);
X = sqrt(W1.^2 + W2.^2);

% ����PDF�Ա�
[counts, edges] = histcounts(X, 50, 'Normalization', 'pdf');
bin_centers = (edges(1:end-1) + edges(2:end))/2;
x_theory = linspace(0, max(X), 1000);
pdf_theory = (x_theory / sigma^2) .* exp(-x_theory.^2 / (2*sigma^2));

figure;
bar(bin_centers, counts); hold on;
plot(x_theory, pdf_theory, 'r', 'LineWidth', 2);
xlabel('����'); ylabel('�����ܶ�');
title(['�����ֲ�����=' num2str(sigma) '��������֤']);
legend('����ֱ��ͼ', '��������');
grid on;

%% ��˹�ֲ���Դ��ģ����֤
clc; clear; close all;

% ��������
N = 1e5;        % ������
K_dB = 10;      % ��˹����K��dB��
sigma = 1;      % ����������׼��

% ������˹����
K = 10^(K_dB/10);
beta = sqrt(2*K*sigma^2);  % ֱ��·������?:ml-citation{ref="3,5" data="citationList"}

% ������˹�ֲ��ź�
W1 = sigma * randn(1, N);
W2 = sigma * randn(1, N);
X = sqrt((beta + W1).^2 + W2.^2); 

% ����PDF�Ա�
[counts, edges] = histcounts(X, 50, 'Normalization', 'pdf');
bin_centers = (edges(1:end-1) + edges(2:end))/2;

% ����PDF����
x_theory = linspace(0, max(X), 1000);
A = beta / sigma;
s = sigma;
pdf_theory = (x_theory / s^2) .* exp(-(x_theory.^2 + A^2*s^2)/(2*s^2)) .* besseli(0, (x_theory*A)/s^2);

% ��ͼ
figure;
bar(bin_centers, counts); hold on;
plot(x_theory, pdf_theory, 'r', 'LineWidth', 2);
xlabel('����'); ylabel('�����ܶ�');
title(['��˹�ֲ���K=' num2str(K_dB) 'dB��������֤']);
legend('����ֱ��ͼ', '��������');
grid on;

%% ���ɷֲ���Դ��ģ����֤
clc; clear; close all;
% ��������
N = 1e5;        % ��������
lambda = 5;     % ƽ��������

% ���ɲ��ɷֲ���Դ
X = poissrnd(lambda, 1, N);    % �������ú������ɲ����������?:ml-citation{ref="4" data="citationList"}

% ����PMF�Ա�
k_values = 0:max(X);
pmf_sim = histcounts(X, [k_values-0.5, max(k_values)+0.5], 'Normalization', 'probability');
pmf_theory = poisspdf(k_values, lambda);

figure;
stem(k_values, pmf_sim, 'bo', 'MarkerSize', 5, 'LineWidth', 1.5); hold on;
plot(k_values, pmf_theory, 'r*', 'LineWidth', 1.5);
xlabel('k'); ylabel('����');
title(['���ɷֲ�����=' num2str(lambda) '��PMF�Ա�']);
legend('��������', '��������');
grid on;

%% ��������ɢ�޼�����Դ
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
fprintf('\n'); % ����
fprintf('����P0x= %.4f\n', P0);
fprintf('����P1x= %.4f\n', P1);
fprintf('ʵ��P0x= %.4f\n', P0x);
fprintf('ʵ��P1x= %.4f\n', P1x);
fprintf('������H= %.4f\n', H1);
fprintf('ʵ����H= %.4f\n', H2);







































