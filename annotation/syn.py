import os
import glob
from moviepy.audio.io.AudioFileClip import AudioFileClip
import numpy as np
from matplotlib import pyplot as plt
import json
from moviepy.editor import VideoFileClip
import scipy.signal as signal
from tqdm import tqdm

XCORR_FIND_ENDTIME = True
VIS_XCORR_FIND_TIMEEND = True
est_delay = 0
fs_motion = 200
fs_audio = 48000
fs_csi = 400

def load_json(filename):
    if (os.path.exists(filename) == False):
        return None
    with open(filename) as f:
        data = f.read()

        input_js = json.loads(data)
        return input_js

def xcorr_find_timeend(videofile, smartphonedata):
    video_audio = VideoFileClip(videofile)
    clip = video_audio.audio
    output_array = clip.to_soundarray()
    audio_signal = smartphonedata['audio_signal']
    fs_video_audio = 44100
    number_of_samples = round(len(audio_signal) * float(fs_video_audio) / fs_audio)
    print(np.shape(audio_signal))
    print(number_of_samples)
    audio_signal = signal.resample(audio_signal, number_of_samples)
    output_array = np.squeeze(output_array[:,0])
    timestart = 20
    timeend = 260
    corr = signal.correlate(output_array[fs_video_audio*timestart:fs_video_audio*timeend], audio_signal[fs_video_audio*timestart:fs_video_audio*timeend])
    corr = corr[int(np.shape(corr)[0]/2): int(np.shape(corr)[0]/2 + 30*fs_video_audio)]
    delay = (np.argmax(corr) - 0) / fs_video_audio
    endtime = np.shape(audio_signal)[0]/fs_video_audio + delay
    
    print("------Xcorr sync egocentric video and smartphone data-----")
    print("Delay:", delay)
    print("Endtime:", endtime)
    
    if (VIS_XCORR_FIND_TIMEEND):
        plt.subplot(311)
        plt.plot(output_array[int((delay + timestart) * fs_video_audio): int((delay + timeend)*fs_video_audio)])
        plt.subplot(312)
        plt.plot(audio_signal[fs_video_audio*timestart:fs_video_audio*timeend])
        plt.subplot(313)
        plt.plot(corr)
        plt.show()
    
    return endtime

def check_correction(smartphonedata, smartphonefile):
    acc_len = np.shape(smartphonedata['acc_matrix'])[0]
    gyro_len = np.shape(smartphonedata['gyro_matrix'])[0]
    csi_len = np.shape(smartphonedata['csi_matrix'])[0]
    audio_len = np.shape(smartphonedata['audio_signal'])[0]
    if (acc_len == gyro_len and acc_len * 2 == csi_len and acc_len * 240 == audio_len):
        print("--------Check Sampling Rate----------")
        print("Pass!")
        return acc_len / 200.0
    else:
        print("Data Sampling Rate Error")
        print("Filename: ", smartphonefile)
        print("Acc: ", str(acc_len))
        print("Gyro: ", str(gyro_len))
        print("CSI: ", str(csi_len))
        print("Audio: ", str(audio_len))
        return 0

def check_datasetnumber(videofilelist, smartphonefilelist):
    print("--------Check Dataset Number----------")
    if (len(videofilelist) != len(smartphonefilelist)):
        print("Dataset number Error!")
        print(videofilelist)
        print(smartphonefilelist)
    else:
        print("Pass!")

def check_videolabel(current_video, current_label):
    print("--------Check Video Labelling----------")
    if (current_video['complete'] == False):
        print("Error: ", current_video['name'], "has not finshed labelling.")
        return 0

    end_enable = 0
    for label_act in current_label:
        if (label_act['label'].lower() == 'end'):
            end_time = label_act['time']
            end_enable += 1
    if (end_enable == 0):
        print("Error: no end label!")
        return 0
    if (end_enable > 1):
        print("Error: multiple end labels!")
        return 0
    
    print("Pass!")
    return end_time

def sync_label(current_label, end_time, time_len):
    print("--------Sync label----------")
    start_time = end_time - est_delay - time_len
    print(start_time)
    print("time_len", time_len)
    current_label_sync = list()
    for label_act in current_label:
        label_act['time'] = label_act['time'] - start_time
        if (label_act['time'] > 0):
            current_label_sync.append(label_act)
    print("Finish!")
    return current_label_sync

def vis(current_label, data, fs, num):
    # index_motion_start = int(current_label[0]['time'] * fs)
    # gap = (np.max(data) - np.min(data)) / num
    # for i in range(num):
    #     act_label = current_label[i]
    #     index_motion_end = int(current_label[i+1]['time'] * fs)
    #     if (index_motion_end > np.shape(data)[0]):
    #         index_motion_end = np.shape(data)[0]
    #     print(index_motion_start, index_motion_end)
    #     plt.plot(range(index_motion_start, index_motion_end), data[index_motion_start: index_motion_end])
    #     plt.text(index_motion_start, i * gap + np.min(data), act_label['label'], fontsize = 8)
    #     index_motion_start = index_motion_end
    # plt.xlabel("Time (s)")
    # values = np.linspace(0,300,num=6)
    # x = np.linspace(0, 300 * fs, num=6)
    # print(values)
    # plt.xticks(x, values)
    # plt.show()

    ## Acc signal
    # plt.specgram(data,Fs=fs,NFFT=128, noverlap=16, mode ='psd', cmap = 'jet')
    # index_motion_start = int(current_label[0]['time'])
    # gap = 100.0 / num
    # for i in range(num):
    #     act_label = current_label[i]
    #     index_motion_end = int(current_label[i+1]['time'])
    #     if (index_motion_end > np.shape(data)[0]):
    #         index_motion_end = np.shape(data)[0]
    #     print(index_motion_start, index_motion_end)
    #     # plt.plot(range(index_motion_start, index_motion_end), data[index_motion_start: index_motion_end])
    #     plt.text(index_motion_start, i * gap + 0, act_label['label'], fontsize = 12)
    #     plt.axvline(x=index_motion_start, ymin=0, ymax=0.9,color ='red',linewidth=1)
    #     index_motion_start = index_motion_end
    # plt.xlim([1, 90])
    # plt.ylim([3, 100])
    # plt.title("Acc")
    # plt.show()

    ## CSI signal
    # plt.specgram(data,Fs=fs,NFFT=256, noverlap=32, mode ='psd', cmap = 'jet')
    # index_motion_start = int(current_label[0]['time'])
    # gap = 200.0 / num
    # for i in range(num):
    #     act_label = current_label[i]
    #     index_motion_end = int(current_label[i+1]['time'])
    #     if (index_motion_end > np.shape(data)[0]):
    #         index_motion_end = np.shape(data)[0]
    #     print(index_motion_start, index_motion_end)
    #     # plt.plot(range(index_motion_start, index_motion_end), data[index_motion_start: index_motion_end])
    #     plt.text(index_motion_start, i * gap + 0, act_label['label'], fontsize = 12)
    #     plt.axvline(x=index_motion_start, ymin=0, ymax=0.9,color ='red',linewidth=1)
    #     index_motion_start = index_motion_end
    # plt.xlim([1, 90])
    # plt.ylim([3, 200])
    # plt.title("WiFi CSI")
    # plt.show()

    ## Audio signal
    plt.specgram(data,Fs=fs,NFFT=1024, noverlap=256, mode ='psd', cmap = 'jet')
    
    index_motion_start = int(current_label[0]['time'])
    if (num != 0):
        gap = 16000.0 / num
        for i in range(num):
            act_label = current_label[i]
            index_motion_end = int(current_label[i+1]['time'])
            if (index_motion_end > np.shape(data)[0]):
                index_motion_end = np.shape(data)[0]
            # print(index_motion_start, index_motion_end)
            # plt.plot(range(index_motion_start, index_motion_end), data[index_motion_start: index_motion_end])
            plt.text(index_motion_start, i * gap + 0, act_label['label'], fontsize = 12)
            plt.axvline(x=index_motion_start, ymin=0, ymax=0.9,color ='red',linewidth=1)
            index_motion_start = index_motion_end
    plt.xlim([1, 300])
    plt.ylim([0, 16000])
    plt.title("Audio Signal")
    plt.colorbar()
    plt.show()


def visualize(smartphonedata, current_label):
    current_label.sort(key = lambda x: (x['time']))
    # vis(current_label, smartphonedata['acc_matrix'][:,0], fs_motion, 10)
    # vis(current_label, smartphonedata['gyro_matrix'][:,0], fs_motion, 40)
    num = len(current_label) - 1
    vis(current_label, smartphonedata['audio_signal'], fs_audio, num)
    # vis(current_label, smartphonedata['csi_matrix'][:,0], fs_csi, 10)
    # smartphonedata, current_label
    
    pass

def check(smartphonefile, labeljson_sync_old):
    global labeljson_sync 
    filename = os.path.split(smartphonefile)[-1][:-4]
    print("*********Check file", filename, "*********")
    smartphonedata = np.load(smartphonefile)
    videofile = os.path.split(smartphonefile)[0][:-10] + "video" + '/' + filename + '.MP4'
    if (labeljson_sync_old != None):
        if (filename + '.MP4' in labeljson_sync_old['data']):
            print("SKIP ...")
            labeljson_sync = labeljson_sync_old
            current_label = labeljson_sync['data'][filename + '.MP4']
            visualize(smartphonedata, current_label)
            return True
            
    labels = labeljson['data']
    videos_label = labeljson['videos']
    current_video = None
    current_label = None
    
    for video_label in videos_label:
        if (video_label['name'] == filename + '.MP4'):
            current_video = video_label
            break
        # print(video_label == os.path.split(smartphonefile)[-1][:-4] + '.MP4')
    if (filename + '.MP4' in labels):
        current_label = labels[filename + '.MP4']
    if (current_video == None or current_label == None):
        print("Error: no corresponding labels in label file!")
        return False
    time_len = check_correction(smartphonedata, smartphonefile)
    if (time_len == 0):
        return False

    if ('sync' not in current_video):
        if (XCORR_FIND_ENDTIME):
            end_time = xcorr_find_timeend(videofile, smartphonedata)
            end_time0 = check_videolabel(current_video, current_label)
            print("Endtime error:", end_time - end_time0)
            str_endtime = input("Endtime (0 estimate, enter automatic):")
        if (str_endtime == ""):
            pass
        elif (str_endtime == '0'):
            print("Endtime estimate: ", end_time0 - 8)
            end_time = end_time0 - 5
        else:
            end_time = time_len + float(str_endtime)
        if (end_time == 0):
            return False
    
        current_label = sync_label(current_label, end_time, time_len)
        current_video['sync'] = True
        labeljson_sync['data'][filename + '.MP4'] = current_label
        labeljson_sync['videos'].append(current_video)


    visualize(smartphonedata, current_label)
    
    return True

def main(filepath, labelfile, labelfile_sync):

    labeljson = load_json(labelfile)
    labeljson_sync = dict()
    labeljson_sync['data'] = dict()
    labeljson_sync['videos'] = list()
    filevideopath = filepath + 'video'
    videofilelist = [f for f in glob.glob(filevideopath + "/*.MP4")]
    videofilelist.sort()

    filesmartphonepath = filepath + 'smartphone'
    smartphonefilelist = [f for f in glob.glob(filesmartphonepath + "/*.npz")]
    smartphonefilelist.sort()


    check_datasetnumber(videofilelist, smartphonefilelist)
    for smartphonefile in tqdm(smartphonefilelist):
        labeljson_sync_old = load_json(labelfile_sync)
        # if (labeljson_sync_old != None):
        #     print(labeljson_sync_old['data'])
        if (not check(smartphonefile, labeljson_sync_old)):
            break
        with open(labelfile_sync, 'w') as f:
            json.dump(labeljson_sync, f)    




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', required=True, type=str, help="File path of dataset")
    parser.add_argument('-l', '--labelfile', required=True, type=str, help="Label file of dataset")
    parser.add_argument('-o', '--labelfilesync', required=True, type=str, help="Output synchronized label file of dataset")
    args = parser.parse_args()

    filepath = args.path
    labelfile = args.labelfile
    labelfile_sync = args.labelfilesync
    main(filepath, labelfile, labelfile_sync)
    
