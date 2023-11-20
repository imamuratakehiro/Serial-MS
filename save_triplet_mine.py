import os
import os.path
import numpy as np
import json
import soundfile as sf
import random
import math
import re

def trackname(no):
    if no in range(1, 10):
        track_name = "Track0000{}".format(no)
    elif no in range(10, 100):
        track_name = "Track000{}".format(no)
    elif no in range(100, 1000):
        track_name = "Track00{}".format(no)
    elif no in range(1000, 2101):
        track_name = "Track0{}".format(no)
    return track_name

def detrackname(name):
    if name[6] != "0":
        no = name[6:]
    elif name[7] != "0":
        no = name[7:]
    elif name[8] != "0":
        no = name[8:]
    elif name[9] != "0":
        no = name[9:]
    return int(no)

def songlist_in_sametampolist(mode):
    sametempolist = json.load(open(f"./metadata/slakh/sametempo_{mode}_edited.json", 'r'))
    songlist = []
    for sametempo in sametempolist:
        for song in sametempo:
            if song in songlist:
                print("error")
            songlist.append(song)
    np.savetxt(f"./metadata/slakh/songlist_in_sametempo_list_{mode}.txt", np.sort(np.array(songlist)), fmt='%d')

def offset2num(file, offset):
    for idx, seg in enumerate(file):
        num = int(seg[1])/44100/offset
        file[idx][1] = int(num)
    return file

class Psuedo:
    def __init__(self, seconds, mode) -> None:
        self.sametempolist = json.load(open(f"./metadata/slakh/sametempo_{mode}_edited.json", 'r'))
        self.datafile = np.loadtxt(f"./metadata/slakh/{seconds}s_no_silence_or0.5_0.25/{mode}_slakh_{seconds}s_stems.txt", delimiter = ",", dtype = "unicode")
        self.datafile = offset2num(self.datafile, offset=seconds/2)
        self.songlist_in_sametempolist = np.loadtxt(f"./metadata/slakh/songlist_in_sametempo_list_{mode}.txt", delimiter = ",", dtype = "unicode")
        self.n_inst = 5
        self.log = []
        self.seconds = seconds
        self.mode = mode
    
    def pick_seg_from_track(self, track):
        segs = np.where(self.datafile[:,0] == trackname(int(track)))[0]
        if len(segs) == 0:
            return -1
        seg = random.choice(segs)
        while seg in self.log:
            seg = random.choice(segs)
        return seg

    def track_exist_checker_in_additional(self, track_id):
        for stlist in self.sametempolist:
            if detrackname(self.datafile[track_id][0]) in stlist:
                return True
        return False
    
    def pick_p_from_a(self, anchor):
        # anchorと同じ曲で前後のセグメントをpositiveとする。overrapを50%しているので2つ前後に
        if anchor+2 >= len(self.datafile):
            return -1
        if self.datafile[anchor+2][0] == self.datafile[anchor][0]:
            #positive = anchor+2
            return anchor+2
        elif self.datafile[anchor-2][0] == self.datafile[anchor][0]:
            #positive = anchor-2
            return anchor-2
        else:
            #fail=True
            return -1
    
    def make_pseudo(self, triposi, sound, a, p, n, a2, p2, n2):
        #print(a, p, n, a2, p2, n2)
        track_a = [detrackname(self.datafile[a][0]) if sound[i] == 1 else -1 for i in range(self.n_inst)]; track_a[triposi] = detrackname(self.datafile[a2][0])
        track_p = [detrackname(self.datafile[n][0]) if sound[i] == 1 else -1 for i in range(self.n_inst)]; track_p[triposi] = detrackname(self.datafile[p2][0])
        track_n = [detrackname(self.datafile[p][0]) if sound[i] == 1 else -1 for i in range(self.n_inst)]; track_n[triposi] = detrackname(self.datafile[n2][0])
        seg_a   = [self.datafile[a][1] if sound[i] == 1 else -1 for i in range(self.n_inst)]; seg_a[triposi] = self.datafile[a2][1]
        seg_p   = [self.datafile[n][1] if sound[i] == 1 else -1 for i in range(self.n_inst)]; seg_p[triposi] = self.datafile[p2][1]
        seg_n   = [self.datafile[p][1] if sound[i] == 1 else -1 for i in range(self.n_inst)]; seg_n[triposi] = self.datafile[n2][1]
        return track_a, track_p, track_n, seg_a, seg_p, seg_n
    
    def adjust_format(self, triplet):
        transformed = []
        for t in triplet:
            tmp = [", ".join([str(x) for x in i]) for i in t] # listの中身をstrに変換してからlist->str変換
            transformed.append("; ".join(tmp))
        return transformed


    def all_streats_pseudo(self, n_song):
        #for i in range(self.cfg.)
        triplet = []
        for i in range(n_song):
            print(i)
            while True:
                fail = False
                # anchor
                a_song = random.choice(self.songlist_in_sametempolist)
                anchor = self.pick_seg_from_track(a_song)
                if anchor == -1:
                    print(f"\tAnchor does not have any segs. : {a_song}")
                    continue
                # positive
                # anchorと同じ曲で前後のセグメントをpositiveとする。overrapを50%しているので2つ前後に
                positive = self.pick_p_from_a(anchor)
                if fail or positive == -1:
                    print(f"\tThere is no same song seg to anchor : {self.datafile[anchor][0]}")
                    #log.append(anchor)
                    continue
                #if not self.track_exist_checker_in_additional(anchor):
                #    print(f"\tanchor not found fail : {self.datafile[anchor]}") # sametempolistに曲がない時
                #    self.log.append(anchor)
                #    continue
                # negative
                # 同じテンポでanchorとは違う曲をnegativeとして3曲取り出す
                for stlist in self.sametempolist:
                    if detrackname(self.datafile[anchor][0]) in stlist:
                        # negativeの曲から、2つから1セグメント、もう1つから隣り合う2セグメント取り出す
                        negative_songs = random.sample(stlist, 3)
                        while int(a_song) in negative_songs: # anchorと曲が被っていたらやり直し
                            negative_songs = random.sample(stlist, 3)
                        n1 = negative_songs[0]; n2 = negative_songs[1]; n3 = negative_songs[2]
                        negative  = self.pick_seg_from_track(n1)
                        anchor2   = self.pick_seg_from_track(n2)
                        negative2 = self.pick_seg_from_track(n3)
                        if negative == -1 or anchor2 == -1 or negative2 == -1:
                            fail = True # 無音排除で曲丸ごとない時
                            break
                        positive2 = self.pick_p_from_a(anchor2)
                        if positive2 == -1:
                            fail = True
                        break
                if fail:
                    print(f"\tnegative not found fail : {trackname(n1)}, {trackname(n2)}, {trackname(n3)}")
                    continue
                break
            c = random.randint(1, 31) # 0(完全無音)はtripletできないので省く
            c = format(c, f"0{self.n_inst}b") #2進数化
            sound = [int(i) for i in c]
            c = np.array(sound)
            # basic,additionalの場所を決めて保存
            place1 = np.where(c==1)[0]
            #print(place1)
            if len(place1) == 1:
                b = place1[0]
                #print(b)
                track_a, track_p, track_n, seg_a, seg_p, seg_n = self.make_pseudo(b, sound, anchor, positive, negative, anchor2, positive2, negative2)
                triplet.append([track_a, track_p, track_n, seg_a, seg_p, seg_n, sound, sound, sound, [b]])
            else:
                b, a = np.random.choice(place1, 2, replace=False)
                #print(b, a)
                track_a, track_p, track_n, seg_a, seg_p, seg_n = self.make_pseudo(b, sound, anchor, positive, negative, anchor2, positive2, negative2)
                triplet.append([track_a, track_p, track_n, seg_a, seg_p, seg_n, sound, sound, sound, [b]])
                triplet.append([track_a, track_n, track_p, seg_a, seg_n, seg_p, sound, sound, sound, [a]])
        triplet = self.adjust_format(triplet)
        with open(f"./metadata/lst/triplets_ba4t_31ways_{self.mode}_{self.seconds}s_{n_song}songs.lst", mode="w") as f:
            f.write("\n".join(triplet))
    
    def songchecker(self, song, n_seg):
        #print(len(np.where(self.datafile[:,0] == trackname(song))[0]))
        if song in self.log:
            return False
        if np.where(self.datafile[:,0] == trackname(song))[0].shape[0] < n_seg:
            #print(np.where(self.datafile[:,0] == trackname(song))[0] , n_seg)
            #print("a")
            return False
        return True

    def pseudo_test(self, n_songs, inst):
        pseudo = []
        n_other_song = 5
        n_seg = 15
        inst_list = ["drums", "bass", "piano", "guitar", "residuals"]
        inst_idx = inst_list.index(inst)
        for s in range(n_songs):
            print(s)
            while True:
                fail = False
                sametempo = random.choice(self.sametempolist)
                if len(sametempo) < n_other_song:
                    continue
                songs = random.sample(sametempo, n_other_song)
                # 既出の曲があったらやり直し、n_seg分曲がなかったらやり直し
                for song in songs:
                    #print(self.songchecker(song, n_seg))
                    if not self.songchecker(song, n_seg):
                        fail = True
                if fail:
                    continue
                #print("a")
                # targetのみで曲をn_seg分
                seg_target_all = np.where(self.datafile[:,0] == trackname(songs[0]))[0]
                seg_target = np.random.choice(seg_target_all, n_seg, replace=False)
                for idx in range(n_seg):
                    track = [songs[0] for i in range(5)]; seg = [self.datafile[seg_target[idx], 1] for i in range(5)]
                    pseudo.append([[inst_idx], track, seg, [songs[0]], [songs[0]]])
                for i in range(1,n_other_song):
                    seg_other_all = np.where(self.datafile[:,0] == trackname(songs[i]))[0]
                    seg_other = np.random.choice(seg_other_all, n_seg, replace=False)
                    for idx in range(n_seg):
                        track = [songs[0] if j == inst_idx else songs[i] for j in range(5)]
                        seg = [self.datafile[seg_target[idx], 1] if j == inst_idx else self.datafile[seg_other[idx], 1] for j in range(5)]
                        pseudo.append([[inst_idx], track, seg, [songs[0]], [songs[i]]])
                for i in range(n_other_song):
                    self.log.append(songs[i])
                break
        pseudo = self.adjust_format(pseudo)
        with open(f"./metadata/lst/psds_{self.mode}_{inst}_{n_songs}songs_{self.seconds}s.lst", mode="w") as f:
            f.write("\n".join(pseudo))




def main():
    pseudo = Psuedo(10, "valid")
    pseudo.pseudo_test(20, "drums")
    pseudo = Psuedo(10, "valid")
    pseudo.pseudo_test(20, "bass")
    pseudo = Psuedo(10, "valid")
    pseudo.pseudo_test(20, "piano")
    pseudo = Psuedo(10, "valid")
    pseudo.pseudo_test(20, "guitar")
    pseudo = Psuedo(10, "valid")
    pseudo.pseudo_test(20, "residuals")
    #pseudo = Psuedo(3, "train")
    #pseudo.all_streats_pseudo(20000)
    #pseudo = Psuedo(5, "train")
    #pseudo.all_streats_pseudo(20000)
    #pseudo = Psuedo(10, "train")
    #pseudo.all_streats_pseudo(20000)
    #pseudo = Psuedo(3, "train")
    #pseudo.all_streats_pseudo(10000)
    #pseudo = Psuedo(5, "train")
    #pseudo.all_streats_pseudo(10000)
    #pseudo = Psuedo(10, "train")
    #pseudo.all_streats_pseudo(10000)
    #pseudo = Psuedo(3, "valid")
    #pseudo.all_streats_pseudo(2000)
    #pseudo = Psuedo(5, "valid")
    #pseudo.all_streats_pseudo(2000)
    #pseudo = Psuedo(10, "valid")
    #pseudo.all_streats_pseudo(2000)
    #pseudo = Psuedo(5, "valid")
    #pseudo.all_streats_pseudo(2000)
    #pseudo = Psuedo(5, "valid")
    #pseudo.all_streats_pseudo(2000)
    #pseudo = Psuedo(5, "valid")
    #pseudo.all_streats_pseudo(2000)
    #songlist_in_sametampolist("test")

if __name__ == "__main__":
    main()