from CSIKit.reader import get_reader
from CSIKit.util import csitools
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import csv
import scipy.io.wavfile as wav
import os
import glob
from tqdm import tqdm

vis_csi = True
vis_audio = False
vis_motion = False

info_csi = True
info_audio = True
info_motion = True
info_sync = True

def resample_ununiformly(x, num, t=None, axis=0):
    resample_signal = np.zeros(num)
    resample_t = np.linspace(t[0], t[-1], num = num, endpoint = True, dtype = float)
    next = 0
    last_signal = x[0]
    next_signal = x[next]
    next_time = t[next]
    for i in range(num):
        if (next == len(t) - 1):
            next_time = t[next]
        else:
            while (next_time < resample_t[i]):
                next += 1
                next_signal = x[next]
                last_signal = x[next - 1]
                next_time = t[next]
        resample_signal[i] = (last_signal + next_signal) / 2.0
    
    return resample_t, resample_signal

def preprocessCSI(csifilename, Fs = 400):
    threshold = 100
    my_reader = get_reader(csifilename)
    csi_data = my_reader.read_file(csifilename)

    ## Extract the csi amplitude
    ## csi_matrix = no_frames * no_subcarriers
    csi_timestamp = csi_data.timestamps
    csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data, metric="amplitude", extract_as_dBm = False, squeeze_output = True)

    ## Extract 80 MHz, 802.11 ac packets
    id_now = 0
    for i in range(no_frames):
        if (np.mean(csi_matrix[i,6:32]) > threshold and csi_matrix[i,32] < threshold and np.mean(csi_matrix[i,69:96]) > threshold and csi_matrix[i,96] < threshold \
            and np.mean(csi_matrix[i,134:160]) > threshold and csi_matrix[i,160] < threshold and np.mean(csi_matrix[i,198:224]) > threshold and csi_matrix[i,224] < threshold):
            csi_matrix[id_now] = csi_matrix[i]
            csi_timestamp[id_now] = csi_timestamp[i]
            id_now += 1

    csi_matrix = csi_matrix[:id_now - 1]
    csi_timestamp = csi_timestamp[:id_now - 1]
    no_frames = id_now

    print("CSI Num of frames before resample:", np.shape(csi_timestamp)[0])

    ## Remove the subcarriers without CSI
    csi_matrix[:,0:26] = csi_matrix[:,6:32]
    csi_matrix[:,26:52] = csi_matrix[:,33:59]
    csi_matrix[:,52:78] = csi_matrix[:,70:96]
    csi_matrix[:,78:104] = csi_matrix[:,97:123]
    csi_matrix[:,104:130] = csi_matrix[:,134:160]
    csi_matrix[:,130:156] = csi_matrix[:,161:187]
    csi_matrix[:,156:182] = csi_matrix[:,198:224]
    csi_matrix[:,182:208] = csi_matrix[:,225:251]
    csi_matrix = csi_matrix[:,:208]
    no_subcarriers = 208

    ## AGC Calibration
    csi_matrix_sum = np.sum(csi_matrix, axis=1)
    csi_matrix = csi_matrix / csi_matrix_sum[:,None]
   
    ### Resample to Fs
    no_frames = int((csi_timestamp[-1] - csi_timestamp[0]) * Fs)
    print(no_frames)
    csi_matrix_resample = np.zeros((no_frames, no_subcarriers), dtype=float)
    csi_matrix_resample_new = np.zeros((no_frames, no_subcarriers), dtype=float)
    csi_timestamp_resample = np.linspace(csi_timestamp[0], csi_timestamp[-1], num = no_frames, endpoint = True, dtype = float)
    for i in range(no_subcarriers):
        resample_t, resample_signal = resample_ununiformly(np.squeeze(csi_matrix[:,i]), no_frames, t = csi_timestamp, axis = 0)
        # a = signal.resample(np.squeeze(csi_matrix[:,i]), no_frames, t = csi_timestamp, axis = 0)
        # csi_matrix_resample[:,i] = a[0]
        csi_matrix_resample[:,i] = resample_signal

    if (vis_csi):
        print(np.shape(csi_matrix_resample))
        print(np.shape(csi_matrix))
        plt.figure()
        plt.subplot(411)
        plt.plot(csi_matrix_resample[:,10])
        # plt.plot(csi_timestamp_resample)
        plt.subplot(412)
        plt.plot(csi_matrix[:,10])
        plt.subplot(413)
        plt.plot(csi_matrix_resample_new[:,10])
        plt.subplot(414)
        plt.plot(csi_timestamp)
        plt.plot(csi_timestamp_resample)
        plt.show()

    if (info_csi):
        print("-----CSI preprocessing information-----")
        print("Num of frames: ", no_frames)
        print("Num of subcarriers: ", no_subcarriers)
        print("Time duration: ", csi_timestamp_resample[-1] - csi_timestamp_resample[0])
    return csi_timestamp_resample, csi_matrix_resample

def preprocessAudio(audiofilename):
    Fs,signal=wav.read(audiofilename)
    signal = signal * 1.0 / 32767
    if (info_audio):
        print("-----Audio preprocessing information-----")
        print("Num of frames: ", np.shape(signal)[0])
        print("Num of subcarriers: ", "Mono")
        print("Time duration: ", np.shape(signal)[0] * 1.0 / Fs)
    if (vis_audio):
        plt.plot(signal)
        plt.show()
    return signal, Fs

def preprocessMotion(motionfilename, Fs = 200):
    motionfile = open(motionfilename)
    csvreader = csv.reader(motionfile)
    csvreader = list(csvreader)[1:]
    no_frames = len(csvreader)
    no_subcarriers = 3
    motion_timestamp = np.zeros((no_frames), dtype = float)
    motion_matrix = np.zeros((no_frames,no_subcarriers), dtype = float)
    # print(no_frames)
    
    for i in range(no_frames):
        row = csvreader[i]
        motion_timestamp[i] = float(row[0]) / 1000.0
        for j in range(no_subcarriers):
            motion_matrix[i][j] = float(row[j+1])

    no_frames = int((motion_timestamp[-1] - motion_timestamp[0]) * Fs)
    motion_matrix_resample = np.zeros((no_frames, no_subcarriers), dtype=float)
    motion_timestamp_resample = np.linspace(motion_timestamp[0], motion_timestamp[-1], num = no_frames, endpoint = True, dtype = float)
    for i in range(no_subcarriers):
        a = signal.resample(np.squeeze(motion_matrix[:,i]), no_frames, t = motion_timestamp, axis = 0)
        motion_matrix_resample[:,i] = a[0]
    if (info_csi):
        print("-----Motion Sensor preprocessing information-----")
        print("Num of frames: ", no_frames)
        print("Num of subcarriers: ", no_subcarriers)
        print("Time duration: ", motion_timestamp_resample[-1] - motion_timestamp_resample[0])
    return motion_timestamp_resample, motion_matrix_resample

def save_npz(filename, acc_matrix, gyro_matrix, csi_matrix, audio_signal):
    np.savez_compressed(filename + '.npz', acc_matrix = acc_matrix, gyro_matrix = gyro_matrix, csi_matrix = csi_matrix, audio_signal = audio_signal)

def preprocess(filename):
    # if os.path.exists(filename + ".npz"):
        # return
    csifilename = filename + '.pcap'
    csi_fs = 400
    csi_timestamp_resample, csi_matrix_resample = preprocessCSI(csifilename, csi_fs)
    # print(csi_timestamp_resample[0], csi_timestamp_resample[-1])

    audiofilename = filename + '.wav'
    audio_signal, audio_fs = preprocessAudio(audiofilename)

    accfilename = filename + '_acc.csv'
    acc_fs = 200
    acc_timestamp_resample, acc_matrix_resample = preprocessMotion(accfilename, acc_fs)
    # print(acc_timestamp_resample[0], acc_timestamp_resample[-1])

    gyrofilename = filename + '_gyro.csv'
    gyro_fs = 200
    gyro_timestamp_resample, gyro_matrix_resample = preprocessMotion(gyrofilename, gyro_fs)

    if (np.shape(acc_timestamp_resample)[0] != np.shape(gyro_timestamp_resample)[0]):
        if (acc_timestamp_resample[-1] < gyro_timestamp_resample[-1]):
            gyro_timestamp_resample = acc_timestamp_resample
            gyro_matrix_resample = gyro_matrix_resample[:np.shape(acc_timestamp_resample)[0]]
        else:
            acc_timestamp_resample = gyro_timestamp_resample
            acc_matrix_resample = acc_matrix_resample[:np.shape(gyro_timestamp_resample)[0]]

    ## Sync three modalities
    time_start = max(csi_timestamp_resample[0], acc_timestamp_resample[0])
    # print(csi_timestamp_resample[-1], acc_timestamp_resample[-1])
    time_end = min(csi_timestamp_resample[-1], acc_timestamp_resample[-1])
    print("CSI:", csi_timestamp_resample[0], ",", csi_timestamp_resample[-1])
    print("Motion:", acc_timestamp_resample[0], ",", acc_timestamp_resample[-1])
    print("Time start:", time_start)
    print("Time end:", time_end)

    start_acc_id = 0
    end_acc_id = 0
    for i in range(len(acc_timestamp_resample)):
        if (acc_timestamp_resample[i] < time_start):
            start_acc_id = i
        if (acc_timestamp_resample[i] <= time_end):
            end_acc_id = i
    start_acc_id += 1

    start_csi_id = 0
    for i in range(len(csi_timestamp_resample)):
        if (csi_timestamp_resample[i] < time_start):
            start_csi_id = i
    start_csi_id += 1
    end_csi_id = start_csi_id + int((end_acc_id - start_acc_id)*(csi_fs / acc_fs))

    start_audio_id = int(start_acc_id * (audio_fs/acc_fs))
    end_audio_id = int(end_acc_id * (audio_fs/acc_fs))
    if (end_audio_id > len(audio_signal)):
        left_audio = end_audio_id - len(audio_signal)
        end_audio_id = len(audio_signal)
        end_acc_id = end_acc_id - int((left_audio) * acc_fs / (audio_fs * 1.0))
        end_csi_id = int(start_csi_id + (end_acc_id - start_acc_id)*(csi_fs / acc_fs))
        end_audio_id = int(end_acc_id * (audio_fs/acc_fs))
    else:
        pass
    audio_signal = audio_signal[start_audio_id:end_audio_id]
    audio_sample_num = np.shape(audio_signal)[0]
    acc_timestamp_resample = acc_timestamp_resample[start_acc_id:end_acc_id]
    acc_matrix_resample = acc_matrix_resample[start_acc_id:end_acc_id]
    gyro_timestamp_resample = gyro_timestamp_resample[start_acc_id:end_acc_id]
    gyro_matrix_resample = gyro_matrix_resample[start_acc_id:end_acc_id]
    # print(start_csi_id, end_csi_id)
    csi_timestamp_resample = csi_timestamp_resample[start_csi_id:end_csi_id]
    csi_matrix_resample = csi_matrix_resample[start_csi_id:end_csi_id]

    #!!!
    save_npz(filename, acc_matrix_resample, gyro_matrix_resample, csi_matrix_resample, audio_signal)
    
    # if (os.path.exists(filename + ".wav")):
    #         os.remove(filename + ".wav")
    # if (os.path.exists(filename + ".pcap")):
    #     os.remove(filename + ".pcap")
    # if (os.path.exists(filename + "_acc.csv")):
    #     os.remove(filename + "_acc.csv")
    # if (os.path.exists(filename + "_gyro.csv")):
    #     os.remove(filename + "_gyro.csv")
    
    if (info_sync):
        print("-----Sensor Sync information-----")
        print("Time duration: ", audio_sample_num * 1.0 / audio_fs)
        print("CSI Num of frames:", np.shape(csi_timestamp_resample)[0])
        print("Acc Num of frames:", np.shape(acc_timestamp_resample)[0])
        print("Gyro Num of frames:", np.shape(gyro_timestamp_resample)[0])
        print("Audio Num of frames:", audio_sample_num)

    # time_start = float(filename.split("/")[-1]) / 1000.0
    # print("CSI:", csi_timestamp_resample[0] - time_start)
    # print("Motion:", motion_timestamp_resample[0] - time_start)
    pass
def check_duration(filename):
    import cv2
    video = cv2.VideoCapture(filename)

    duration = video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    return duration

def compress_video(filename):
    fsize = os.path.getsize(filename)
    duration = check_duration(filename)
    print(fsize/duration)
    if (fsize/duration < 1000000):
        return
    cmd = "ffmpeg -i " + filename + " -vf scale=960:720 -c:v libx264 " + filename[:-4] + "_compress.MP4" + " -hide_banner"
    os.system(cmd)
    os.remove(filename)
    os.rename(filename[:-4] + "_compress.MP4", filename)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', required=True, type=str, help="File path of dataset")
    args = parser.parse_args()

    filevideopath = os.path.join(args.path, "dataset", "video")
    videofilelist = [f for f in glob.glob(os.path.join(filevideopath, "*.MP4"))]
    videofilelist.sort()
    for filename in videofilelist:
        compress_video(filename)

    filesmartphonepath = os.path.join(args.path, "dataset", "smartphone")
    smartphonefilelist = [f for f in glob.glob(os.path.join(filesmartphonepath + "*.wav"))]
    smartphonefilelist.sort()
    for filename in tqdm(smartphonefilelist):
        filename = filename[:-4]
        print(filename)
        preprocess(filename)
    