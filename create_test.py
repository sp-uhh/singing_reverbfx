import csv
import sys
from argparse import ArgumentParser
from os import makedirs
from os.path import exists, join

import numpy as np
from librosa import resample
from scipy import stats
from scipy.signal import convolve
from soundfile import read, write
from tqdm import tqdm


def save_files(target_dir, subset, speaker, id, vocal_file, rir_file, channel, 
               gain, rt60, mixture, vocal, args, dry_wet):

    with open(join(target_dir, f"{subset}.csv"), "a") as text_file:
        text_file.write(f"{id:05},{speaker},{vocal_file},"
            + f"{rir_file.replace(args.data_dir, '')},{channel},{gain},{rt60:.2f},{dry_wet}\n")
        
    #removed speaker directory, because of no singer information  
    write(join(target_dir, subset, "reverberant", f"{id:05}.wav"), mixture, args.target_sr, subtype="FLOAT")
    write(join(target_dir, subset, "clean", f"{id:05}.wav"), vocal, args.target_sr, subtype="FLOAT")

def calc_rt60(h, sr=48000, rt='t30'): 
    """
    RT60 measurement routine acording to Schroeder's method [1].

    [1] M. R. Schroeder, "New Method of Measuring Reverberation Time," J. Acoust. Soc. Am., vol. 37, no. 3, pp. 409-412, Mar. 1968.

    Adapted from https://github.com/python-acoustics/python-acoustics/blob/99d79206159b822ea2f4e9d27c8b2fbfeb704d38/acoustics/room.py#L156
    """
    rt = rt.lower()
    if rt == 't30':
        init = -5.0
        end = -35.0
        factor = 2.0
    elif rt == 't20':
        init = -5.0
        end = -25.0
        factor = 3.0
    elif rt == 't10':
        init = -5.0
        end = -15.0
        factor = 6.0
    elif rt == 'edt':
        init = 0.0
        end = -10.0
        factor = 6.0

    h_abs = np.abs(h) / np.max(np.abs(h))

    # Schroeder integration
    sch = np.cumsum(h_abs[::-1]**2)[::-1]
    sch_db = 10.0 * np.log10(sch / np.max(sch)+1e-20)

    # Linear regression
    sch_init = sch_db[np.abs(sch_db - init).argmin()]
    sch_end = sch_db[np.abs(sch_db - end).argmin()]
    init_sample = np.where(sch_db == sch_init)[0][0]
    end_sample = np.where(sch_db == sch_end)[0][0]
    x = np.arange(init_sample, end_sample + 1) / sr
    y = sch_db[init_sample:end_sample + 1]
    slope, intercept = stats.linregress(x, y)[0:2]

    # Reverberation time (T30, T20, T10 or EDT)
    db_regress_init = (init - intercept) / slope
    db_regress_end = (end - intercept) / slope
    t60 = factor * (db_regress_end - db_regress_init)
    return t60


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help='Path to data directory which should contain subdirectories EARS and WHAM!48kHz')
    parser.add_argument("--target_sr", type=int, default=48000, help='Sampling rate')
    args = parser.parse_args()

    # Organize directories
    target_dir = join(args.data_dir, "SingingReverbFX")

    # Load meta data
    with open('test.csv', 'r', newline='') as f:
        reader = csv.DictReader(f)  # Each row becomes a dict keyed by column names
        rows = list(reader)

    # NHSS (https://hltnus.github.io/NHSSDatabase/)
    vocal_dir = join(args.data_dir, "NHSS")
    
    if exists(target_dir):
        print(f"[Warning] Abort Singing-Reverb generation script. The directory {join(args.data_dir, target_dir)} already exists.")
        sys.exit()
    else:
        makedirs(target_dir)

    rir_test_dir = join(args.data_dir, "ReverbFX", "test")
    subset = "test"
    
    print(f"Generate {subset} split")

    with open(join(target_dir, f"{subset}.csv"), "w") as text_file:
        text_file.write(f"id,speaker,vocal_file,rir_file,channel,gain,rt60,dry_wet\n")

    makedirs(join(target_dir, subset, "clean"), exist_ok=True)
    makedirs(join(target_dir, subset, "reverberant"), exist_ok=True)

    for row in tqdm(rows):

        id = int(row['id'])
        vocal_file = join(vocal_dir, row['vocal_file'])
        rir_file = join(rir_test_dir, row['rir_file'])
        channel = int(row['channel'])
        gain = float(row['gain'])
        dry_wet = float(row['dry_wet'])

        vocal, sr = read(vocal_file)

        # Mearge channels to mono
        if vocal.ndim > 1:
            vocal = np.mean(vocal, axis=1)

        # Resample if necessary
        if sr != args.target_sr:
            vocal = resample(vocal.astype(float), orig_sr=sr, target_sr=args.target_sr)
            
        # Sample RIRs until RT60 is below max_rt60 and pre_samples are below max_pre_samples

        rir, sr = read(rir_file, always_2d=True)
        # Take mono channel

        rir = rir[:,channel]

        if sr != args.target_sr:
            rir = resample(rir.astype(float), orig_sr=sr, target_sr=args.target_sr)

        # Cut RIR to get direct path at the beginning
        max_index = np.argmax(np.abs(rir))
        rir = rir[max_index:]

        # Normalize RIRs in range [0.1, 0.7]
        if np.max(np.abs(rir)) < 0.1:
            rir = 0.1 * rir / np.max(np.abs(rir))
        elif np.max(np.abs(rir)) > 0.7:
            rir = 0.7 * rir / np.max(np.abs(rir))

        rt60 = calc_rt60(rir, sr=args.target_sr)

        mixture = convolve(vocal, rir)[:len(vocal)]

        #Dry Wet
        mixture = (1-dry_wet) * vocal + dry_wet * mixture
        
        # Normalize mixture
        mixture = gain * mixture

        if np.max(np.abs(mixture)) > 1.0:
            mixture = mixture / np.max(np.abs(mixture))

        save_files(target_dir, subset, ".", id, vocal_file, rir_file,
                        channel, gain, rt60, mixture, vocal, args, dry_wet)
