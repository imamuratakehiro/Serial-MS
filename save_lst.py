import pandas as pd
import numpy as np

def comma_semicoron(i, trg_seg, oth_seg, id, id_oth):
    trg_seg = ", ".join([str(x) for x in trg_seg])
    oth_seg = ", ".join([str(x) for x in oth_seg])
    return f"{i}; {trg_seg}; {oth_seg}; {id}, {id_oth}"


def test_lst_id_ver():
    seg_list = pd.read_csv("./metadata/zume/slakh/10s_on10.0_no_silence_all.csv", index_col=0).values
    
    track_id = [
        1883,
        1936,
        1940,
        1950,
        2004,
        2016,
        2047,
        2050,
        2051,
        2052,]
    """
    track_id = [
        1510,
        1520,
        1582,
        1595,
        1620,
        1624,
        1641,
        1642,
        1643,
        1645,
    ]
    """
    for id in track_id:
        print(np.where(seg_list[:,0] == id)[0])
    inst_list = ["drums", "bass", "piano", "guitar", "residuals"]
    for i, inst in enumerate(inst_list):
        inst_psd = []
        for id in track_id:
            track_trg_seg = np.random.choice(np.where(seg_list[:,0] == id)[0], 10)
            for id_oth in track_id: #1曲に対して10曲作成を10曲で
                track_oth_seg = np.random.choice(np.where(seg_list[:,0] == id_oth)[0], 10)
                for trg_seg, oth_seg in zip(track_trg_seg, track_oth_seg):
                    psd_track = []
                    psd_seg = []
                    for j in range(len(inst_list)):
                        if i == j:
                            psd_track.append(id); psd_seg.append(seg_list[trg_seg, 1])
                        else:
                            psd_track.append(id_oth); psd_seg.append(seg_list[oth_seg, 1])
                    inst_psd.append(comma_semicoron(i, psd_track, psd_seg, id, id_oth))
                print(id, trg_seg, id_oth, oth_seg)
        with open(f"./metadata/lst/psd_test_{inst}_10_10.lst", mode="w") as f:
            for item in inst_psd:
                f.write("{}\n".format(item))


def main():
    test_lst_id_ver()

if __name__ == "__main__":
    main()
