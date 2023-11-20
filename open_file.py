import numpy as np
import os
import pandas as pd

def loadseg_from_npz(path):
    npz = np.load(path)
    return npz["wave"].astype(np.float32), npz["sound"]

def read_file_and_write_down(file):
    path = f"/nas03/assets/Dataset/slakh/cutwave/{file}/bass"
    files = os.listdir(path)
    songseg_list = []
    for id in range(1, 2101):
        print(id)
        counter = 0
        while True:
            if f"wave{id}_{counter}.npz" in files:
                songseg_list.append([id, counter])
                counter += 1
            else:
                break
    df = pd.DataFrame(songseg_list, columns=["track_id", "seg_id"])
    df.to_csv(f"./metadata/zume/slakh/{file}.csv")

def read_file_and_write_down_no_silence_all(file):
    songseg_list = []
    datafile = pd.read_csv(f"./metadata/zume/slakh/{file}.csv", index_col=0).values
    for track in datafile:
        sound_all = True
        for inst in ["drums", "bass", "piano", "guitar", "residuals"]:
            path = f"/nas03/assets/Dataset/slakh/cutwave/{file}/{inst}/wave{track[0]}_{track[1]}.npz"
            _, sound = loadseg_from_npz(path)
            sound_all = sound_all and sound
        if sound_all:
            print(track[0], track[1])
            songseg_list.append([track[0], track[1]])
    df = pd.DataFrame(songseg_list, columns=["track_id", "seg_id"])
    df.to_csv(f"./metadata/zume/slakh/{file}_no_silence_all.csv")

def read_file_and_write_down2():
    path = f"/nas03/assets/Dataset/slakh/single3_200data-euc_zero/bass/"
    #files = os.listdir(path)
    songseg_list = []
    for id in range(1, 205):
        if os.path.isdir(path + f"Track{id}"):
            files = os.listdir(path + f"Track{id}")
        else:
            continue
        print(id)
        counter = 0
        while True:
            if f"seg{counter}.npy" in files:
                songseg_list.append([id, counter])
                counter += 1
            else:
                break
    df = pd.DataFrame(songseg_list, columns=["track_id", "seg_id"])
    df.to_csv(f"./metadata/zume/slakh/single3_200data-euc_zero.csv")

def main():
    path = "/nas03/assets/Dataset/slakh/single3_200data-euc_zero/bass/Track2/seg0.npy"
    #print(np.load(path))
    file = "3s_on1.5"
    read_file_and_write_down_no_silence_all(file)
    file = "10s_on5.0"
    read_file_and_write_down_no_silence_all(file)
    file = "10s_on10.0"
    read_file_and_write_down_no_silence_all(file)
    #read_file_and_write_down2()
    #path = "/nas03/assets/Dataset/slakh/cutwave/3s_on1.5/bass/wave1_0.npz"
    #npz = np.load(path)
    #print(npz.files)
    #print(npz["sound"])
    #print(npz["thres"])

if __name__ == "__main__":
    main()