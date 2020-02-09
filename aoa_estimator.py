import argparse
import math
import os
import numpy as np
import noisereduce as nr
import librosa
from scipy import signal

C0 = 343  # 声速
D = 0.1  # 麦克风间距
NOISE_LEN = 4000  # 噪声样本的长度
SR_MULTIPLIER = 16  # 升采样的倍数
T_NEIGHBOR = 2 * D / C0  # 寻找相关最大值的时间半径（最大可能延时的两倍）
BANDPASS_FILTER = signal.firwin(
    1024, [0.02, 0.3], pass_zero=False)  # 带通 FIR 滤波器


def read_audio(filename):
    y, sr = librosa.load(filename, sr=None, mono=False)
    ch1 = y[0, :]
    ch2 = y[1, :]
    return sr, ch1, ch2


def reduce_noise(ch1, ch2, noise_len):
    ''' 去噪 '''
    ch1_noise = ch1[0:noise_len]
    ch1_dn = nr.reduce_noise(audio_clip=ch1, noise_clip=ch1_noise, n_grad_freq=2,
                             n_grad_time=6, n_fft=8192, win_length=8192,
                             hop_length=128, n_std_thresh=1.5, prop_decrease=1)
    ch2_noise = ch2[0:noise_len]
    ch2_dn = nr.reduce_noise(audio_clip=ch2, noise_clip=ch2_noise, n_grad_freq=2,
                             n_grad_time=6, n_fft=8192, win_length=8192,
                             hop_length=128, n_std_thresh=1.5, prop_decrease=1)
    return ch1_dn, ch2_dn


def resample(ch1, ch2, orig_sr, target_sr):
    ''' 变换采样率 '''
    ch1_new = librosa.resample(ch1, orig_sr, target_sr)
    ch2_new = librosa.resample(ch2, orig_sr, target_sr)
    return ch1_new, ch2_new


def calc_relevance(ch1, ch2):
    ''' 计算相关函数 '''
    n_sample = len(ch1)
    n_fft = 2 ** math.ceil(math.log2(2 * n_sample - 1))
    CH1 = np.fft.fft(ch1, n_fft)
    CH2 = np.fft.fft(ch2, n_fft)
    G = np.multiply(CH1, np.conj(CH2))
    r = np.fft.fftshift(np.real(np.fft.ifft(G, n_fft)))
    return r


def calc_angle(delta_n, sr, c0, d):
    ''' 估计声源角度 '''
    delta_t = delta_n / sr
    cos_theta = c0 * delta_t / d

    # 非线性映射，减小两端的估计误差
    if cos_theta > 0.995:
        theta = math.acos(1 - 10.92 * math.exp(24.17 - 32.16 * cos_theta))
    elif cos_theta < -0.995:
        theta = math.acos(-1 + 10.92 * math.exp(24.17 + 32.16 * cos_theta))
    else:
        theta = math.acos(cos_theta)

    theta_degree = theta / math.pi * 180
    return theta_degree


def bandpass_filter(ch1, ch2):
    ''' 去除信号的低频、高频分量 '''
    ch1_ft = signal.lfilter(BANDPASS_FILTER, [1.0], ch1)
    ch2_ft = signal.lfilter(BANDPASS_FILTER, [1.0], ch2)
    return ch1_ft, ch2_ft


def estimate(ch1, ch2, sr):
    ''' 估计声源的角度 '''
    # 低切、高切
    ch1_ft, ch2_ft = bandpass_filter(ch1, ch2)
    # 去噪
    ch1_dn, ch2_dn = reduce_noise(ch1_ft, ch2_ft, NOISE_LEN)
    # 升采样
    sr_up = sr * SR_MULTIPLIER
    ch1_up, ch2_up = resample(ch1_dn, ch2_dn, sr, sr_up)
    # 求角度
    r = calc_relevance(ch1_up, ch2_up)
    n_mid = int(len(r) / 2)
    n_neighbor = int(sr_up * T_NEIGHBOR)
    delta_n = np.argmax(
        r[n_mid - n_neighbor: n_mid + n_neighbor]) - n_neighbor
    return calc_angle(delta_n, sr_up, C0, D)


def main():
    parser = argparse.ArgumentParser(
        description='''This tool will estimate the Angle of Arrival (AoA) of the sound source from audio files.''')
    parser.add_argument('-d', '--directory', dest='directory', required=True,
                        help='the parent directory of audio files')
    parser.add_argument('-n', '--number', dest='number', required=True,
                        help='the number of audio files')

    args = parser.parse_args()
    number = int(args.number)
    print('Start processing:\n')

    angles = np.zeros(number)
    for k in range(1, number + 1):
        path = os.path.join(args.directory, str(k) + '.wav')
        sr, ch1, ch2 = read_audio(path)
        angles[k - 1] = estimate(ch1, ch2, sr)
        print('{:6d}: Estimated Angle is {:.2f} degree'.format(
            k, angles[k - 1]))

    out_path = os.path.join(args.directory, 'result.txt')
    print('\nAll jobs finished! Writing results to "{}" ...\n'.format(out_path))

    with open(out_path, 'w') as f:
        f.writelines(["{}\n".format(item) for item in angles])


if __name__ == '__main__':
    main()
