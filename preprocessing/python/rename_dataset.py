# copy this file to parent directory of dataset_new
# expect a directory called dataset to be created

import glob
import os
import shutil

def pairsmartphonevideo(filepath, videopath_new, smartphonepath_new, destination_path='dataset'):
    videofilelist = [f for f in glob.glob(os.path.join(videopath_new, "*.MP4"))]
    videofilelist.sort()
    # print(videofilelist)
    smartphonefilelist = [f for f in glob.glob(os.path.join(smartphonepath_new, "*.wav"))]
    smartphonefilelist.sort()
    # print(smartphonefilelist)

    if len(videofilelist) != len(smartphonefilelist):
        print("Dataset_new Import Error!")
        print(videofilelist)
        print(smartphonefilelist)
        return

    destination = os.path.join(filepath, destination_path)
    if not os.path.exists(destination):
        os.mkdir(destination)
    videopath = os.path.join(destination, 'video')
    if not os.path.exists(videopath):
        os.mkdir(videopath)
    smartphonepath = os.path.join(destination, 'smartphone')
    if not os.path.exists(smartphonepath):
        os.mkdir(smartphonepath)

    # move MP4
    for i in range(len(videofilelist)):
        base_name = os.path.splitext(os.path.basename(smartphonefilelist[i]))[0]
        print("Moving file: ", base_name)
        shutil.move(videofilelist[i], os.path.join(videopath, base_name + '.MP4'))

    # move all files in smartphone directory directly
    for file_name in os.listdir(smartphonepath_new):
        shutil.move(os.path.join(smartphonepath_new, file_name), smartphonepath)
    
    print("Moving finished. Please manually check dataset_new directory")

    return

def main(filepath):
    filesmartphonepath_new = os.path.join(filepath, 'dataset_new', 'smartphone')
    filevideopath_new = os.path.join(filepath, 'dataset_new', 'video')
    pairsmartphonevideo(filepath, filevideopath_new, filesmartphonepath_new)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', required=True, type=str, help="File path of dataset")
    args = parser.parse_args()
    main(args.path)