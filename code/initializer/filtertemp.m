clc; clear;
f1=100;f2=200;%待滤波正弦信号频率
fs=2000;%采样频率
m=(0.3*f1)/(fs/2);%定义过度带宽
M=round(8/m);%定义窗函数的长度
N=M-1;%定义滤波器的阶数
b=fir1(N,0.5*f2/(fs/2));%使用fir1函数设计滤波器
%输入的参数分别是滤波器的阶数和截止频率
figure(1)
[h,f]=freqz(b,1,512);%滤波器的幅频特性图
%[H,W]=freqz(B,A,N)当N是一个整数时函数返回N点的频率向量和幅频响应向量
plot(f*fs/(2*pi),20*log10(abs(h)))%参数分别是频率与幅值
xlabel('频率/赫兹');ylabel('增益/分贝');title('滤波器的增益响应');
figure(2)
subplot(211)
t=0:1/fs:0.5;%定义时间范围和步长
s=sin(2*pi*f1*t)+sin(2*pi*f2*t);%滤波前信号
plot(t,s);%滤波前的信号图像
xlabel('时间/秒');ylabel('幅度');title('信号滤波前时域图');
subplot(212)
Fs=fft(s,512);%将信号变换到频域
AFs=abs(Fs);%信号频域图的幅值
f=(0:255)*fs/512;%频率采样
plot(f,AFs(1:256));%滤波前的信号频域图
xlabel('频率/赫兹');ylabel('幅度');title('信号滤波前频域图');
figure(3)
sf=filter(b,1,s);%使用filter函数对信号进行滤波
%参数分别为滤波器系统函数的分子和分母多项式系数向量和待滤波信号输入
subplot(211)
plot(t,sf)%滤波后的信号图像
xlabel('时间/秒');ylabel('幅度');title('信号滤波后时域图');
axis([0.2 0.5 -2 2]);%限定图像坐标范围
subplot(212)
Fsf=fft(sf,512);%滤波后的信号频域图
AFsf=abs(Fsf);%信号频域图的幅值
f=(0:255)*fs/512;%频率采样
plot(f,AFsf(1:256))%滤波后的信号频域图
xlabel('频率/赫兹');ylabel('幅度');title('信号滤波后频域图');
