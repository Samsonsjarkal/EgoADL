import uuid
from tqdm import tqdm
import os
import json
import numpy as np
import csv
import re

MOTION_FS  = 200
CSI_FS = 400
AUDIO_FS = 48000
PREPROCESS = True

# "ID": IDs,
# "duration": duration,
# "npz": npz,
# "semantics": semantics,
# "transcript": transcript,

def load_json(filename):
    with open(filename) as f:
        data = f.read()
    input_js = json.loads(data)
    return input_js

def save_npz(filename, acc_matrix, gyro_matrix, csi_matrix, audio_signal):
    np.savez_compressed(filename + '.npz', acc_matrix = acc_matrix, gyro_matrix = gyro_matrix, csi_matrix = csi_matrix, audio_signal = audio_signal)

def exactuser(npzfile):
    print(npzfile)
    user = npzfile.split('/')[4]
    user = user.lower() 
    # re.sub(r'[^a-zA-Z]', '', user.lower())
    return user

def check_csi(csi_matrix):
    csi_len = np.shape(csi_matrix)[0]
    num = 0
    for i in range(csi_len - 1):
        if (csi_matrix[i][0] == csi_matrix[i+1][0]):
            num += 1

    rate = num * 1.0 / (csi_len * 1.0)
    # print(rate)
    return rate
    # if (rate > 0.2):
    #     return True
    # else:
    #     return False

def generate_dataset_list_file(label_list, npzfile, filefolder_dataset_seg, csv_filename, segment_context):
    data = np.load(npzfile)
    dataset = list()
    user = exactuser(npzfile)
    for i in tqdm(range(len(label_list))):
        label = label_list[i]
        label_i = re.sub(r'[^a-zA-Z0-9// ]', '', label['label'].lower())
        if (label_i == 'end'):
            break
        starttime = label['time'] - segment_context[0]
        timelen = np.shape(data['audio_signal'])[0] / AUDIO_FS
        if (i < len(label_list) - 1):
            endtime = label_list[i+1]['time'] + 1
        else:
            endtime = timelen
        if (starttime < 0):
            continue
        if (starttime > timelen):
            continue
        if (endtime > timelen):
            endtime = timelen
        ## Timelen <= 10 s
        if (endtime - starttime > 10):
            endtime = starttime + 10
        if (endtime - starttime < 2):
            continue
        id = str(uuid.uuid3(uuid.NAMESPACE_DNS, str(label)))
        # print(id, starttime, endtime, label_i)

        startacc = int(starttime * MOTION_FS)
        endacc = int(endtime * MOTION_FS)
        acc_matrix = data['acc_matrix'][startacc:endacc]

        gyro_matrix = data['gyro_matrix'][startacc:endacc]
        startcsi = int(starttime * CSI_FS)
        endcsi = int(startcsi + (endacc - startacc) * (CSI_FS/MOTION_FS))
        csi_matrix = data['csi_matrix'][startcsi:endcsi]
        rate = check_csi(csi_matrix)
        # print(rate)
        # if (rate > 0.2):
        #     continue

        startaudio = int(starttime * AUDIO_FS)
        endaudio = int(startaudio + (endacc - startacc) * (AUDIO_FS/MOTION_FS))
        audio_signal = data['audio_signal'][startaudio:endaudio]

        duration_acc = np.shape(acc_matrix)[0] * 1.0 / MOTION_FS
        duration_gyro = np.shape(gyro_matrix)[0] * 1.0 / MOTION_FS
        duration_csi = np.shape(csi_matrix)[0] * 1.0 / CSI_FS
        duration_audio = np.shape(audio_signal)[0] * 1.0 / AUDIO_FS
        duration = min([duration_acc, duration_gyro, duration_csi, duration_audio])

        acc_matrix = acc_matrix[:int(duration * MOTION_FS)]
        gyro_matrix = gyro_matrix[:int(duration * MOTION_FS)]
        csi_matrix = csi_matrix[:int(duration * MOTION_FS) * 2]
        audio_signal = audio_signal[:int(duration * MOTION_FS) * 240]
        if (duration <= 0):
            continue

        filename = os.path.join(filefolder_dataset_seg, id)
        if (PREPROCESS):
            save_npz(filename, acc_matrix, gyro_matrix, csi_matrix, audio_signal)

        label_now = list()
        label_now.append(id)
        label_now.append(duration)
        label_now.append(rate)
        label_now.append(filename + '.npz')
        label_now.append(label_i)
        label_now.append(npzfile)
        label_now.append(user)
        label_now.append(starttime)
        dataset.append(label_now)
        # print(label_now)
    
    with open(csv_filename, 'a') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(dataset)

def generate_dataset_list(folder, label, filefolder_dataset_seg, csv_filename):
    data_list = label['data']
    for videoname in data_list:
        npzname = os.path.join(folder, 'smartphone')
        npzname = os.path.join(npzname, videoname[:-4] + '.npz')
        print(videoname)
        generate_dataset_list_file(data_list[videoname], npzname, filefolder_dataset_seg, csv_filename, segment_context = [2, 1])
    pass

def segment_folder(folder, filefolder_dataset_seg, csv_filename):

    ## Read labels
    labelpath = os.path.join(folder, 'label/')
    for labelfile in os.listdir(labelpath):
        if ('clean') in labelfile:
            break
    labelfile = os.path.join(labelpath, labelfile)
    label = load_json(labelfile)
    generate_dataset_list(folder, label, filefolder_dataset_seg, csv_filename)
    # print(label)
    pass

def main(folder, filefolder_dataset_seg, csv_filename):
    # for folder in folderlist:
    print("-------Segmenting", folder, "-----------")
    segment_folder(folder, filefolder_dataset_seg, csv_filename)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', required=True, type=str, help="File path of dataset")
    parser.add_argument('-o', '--outputpath', required=True, type=str, help="Output path of dataset")
    parser.add_argument('-o', '--outputcsv', required=True, type=str, help="Output path of csv filename")
    args = parser.parse_args()
    folder = args.path()
    filefolder_dataset_seg = args.outputpath()
    csv_filename = args.outputcsv()
    
    main(folder, filefolder_dataset_seg, csv_filename)