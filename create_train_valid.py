import csv
import sys
from argparse import ArgumentParser
from glob import glob
from os import makedirs
from os.path import exists, join

import numpy as np
import pyloudnorm as pyln
from librosa import resample
from scipy import stats
from scipy.signal import convolve
from soundfile import read, write
from tqdm import tqdm


def save_files(target_dir, subset, speaker, id, vocal_file, vocal_start, vocal_end, rir_file, channel, 
               gain, rt60, mixture, vocal, args, dry_wet):

    #Put path in string, since some paths contains , 
    relative_vocal_path = f'"{vocal_file}"'

    with open(join(target_dir, f"{subset}.csv"), "a") as text_file:
        text_file.write(f"{id:05},{speaker},{relative_vocal_path},{vocal_start},{vocal_end},"
            + f"{rir_file.replace(args.data_dir, '')},{channel},{gain},{rt60:.2f},{dry_wet}\n")
        
    #removed speaker directory, because of no singer information  
    write(join(target_dir, subset, "reverberant", f"{id:05}_{rt60:.2f}.wav"), mixture, args.target_sr, subtype="FLOAT")
    write(join(target_dir, subset, "clean", f"{id:05}.wav"), vocal, args.target_sr, subtype="FLOAT")
    id += 1
    return id

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

def log_skipped_file(id, file_path, reason, details=""):
        with open(skipped_files_csv, "a", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([id, file_path, reason, details])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help='Path to data directory which should contain subdirectories EARS and WHAM!48kHz')
    parser.add_argument("--min_length", type=float, default=2.0, help='Minimum length of speech files in seconds')
    parser.add_argument("--cut_length", type=float, default=10.0, help='Cut long files to this length in seconds')
    parser.add_argument("--target_sr", type=int, default=48000, help='Sampling rate')
    args = parser.parse_args()

    # Reproducibility
    np.random.seed(42)

    # Organize directories
    target_dir = join(args.data_dir, "SingingReverbFX")

    # Load Vocal datasets
    train_vocal_path = []

    # CSD (https://zenodo.org/records/4785016/files/CSD.zip)
    dir = join(args.data_dir, "CSD")
    train_vocal_path += sorted(glob(join(dir, '**', 'wav', '*.wav'), recursive=True))

    # m4singer (https://drive.google.com/file/d/1xC37E59EWRRFFLdG3aJkVqwtLDgtFNqW/view)
    dir = join(args.data_dir, "m4singer")
    train_vocal_path += sorted(glob(join(dir, '**', '*.wav'), recursive=True))

    # PJS (https://drive.google.com/file/d/1hPHwOkSe2Vnq6hXrhVtzNskJjVMQmvN_/view)
    dir = join(args.data_dir, "PJS_corpus_ver1.1")
    train_vocal_path += sorted(glob(join(dir, '**', '*song.wav'), recursive=True))   

    # OpenSinger (https://drive.google.com/file/d/1EofoZxvalgMjZqzUEuEdleHIZ6SHtNuK/view)
    dir = join(args.data_dir, "OpenSinger")
    train_vocal_path += sorted(glob(join(dir, '**', '*.wav'), recursive=True))

    # Opencpop (https://wenet.org.cn/opencpop/download/)  
    dir = join(args.data_dir, "Opencpop")  
    train_vocal_path += sorted(glob(join(dir, '*.wav'), recursive=True))
    
    # NUS48-e (https://drive.google.com/drive/folders/12pP9uUl0HTVANU3IPLnumTJiRjPtVUMx)
    dir = join(args.data_dir, "nus-smc-corpus_48")
    train_vocal_path += sorted(glob(join(dir, '**', 'sing', '*.wav'), recursive=True))

    #Validation 
    all_opensinger = []
    valid_vocal_path = []
    dir = join(args.data_dir, "OpenSinger")
    valid_vocal_path += sorted(glob(join(dir, '**', '5*.wav'), recursive=True))   
    # Take only files, which start with "5" to get unique man and female singers in the validation set, 980 files in total
    
    # Remove validation paths from train paths
    train_vocal_path = list(set(train_vocal_path) - set(valid_vocal_path))
    
    print(f"Number train files: {len(train_vocal_path)}") 
    print(f"Number valid files: {len(valid_vocal_path)}") 

    if exists(target_dir):
        print(f"[Warning] Abort Singing-Reverb generation script. The directory {join(args.data_dir, target_dir)} already exists.")
        sys.exit()
    else:
        makedirs(target_dir)

    # Error-csv for not processed files 
    skipped_files_csv = join(target_dir, "skipped_files.csv")
    with open(skipped_files_csv, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id","vocal_file", "reason", "details"])

    # Load RIR dataset
    rir_files = []
    rir_train_dir = join(args.data_dir, "ReverbFX", "train")
    rir_train_files = sorted(glob(join(rir_train_dir, "*.wav"), recursive=True))

    rir_valid_dir = join(args.data_dir, "ReverbFX", "valid")
    rir_valid_files = sorted(glob(join(rir_valid_dir, "*.wav"), recursive=True))

    meter = pyln.Meter(args.target_sr)
    
    # Select speech files for split
    for subset, files in zip(["train", "valid"], [train_vocal_path, valid_vocal_path]):
        print(f"Generate {subset} split")

        if subset == "train":
            rir_files = rir_train_files
        elif subset == "valid":
            rir_files = rir_valid_files

        with open(join(target_dir, f"{subset}.csv"), "w") as text_file:
            text_file.write(f"id,speaker,vocal_file,speech_start,speech_end,rir_file,channel,gain,rt60,dry/wet\n")

        makedirs(join(target_dir, subset, "clean"))
        makedirs(join(target_dir, subset, "reverberant"))

        id = 0

        for vocal_file in tqdm(files):
            vocal, sr = read(vocal_file)

            # Only take speech files that are longer than min_length
            if len(vocal) < args.min_length*args.target_sr:
                log_skipped_file(id, vocal_file, "Audio to short", f"{len(vocal) / sr:.2f} seconds")
                continue

            # Mearge channels to mono
            if vocal.ndim > 1:
                vocal = np.mean(vocal, axis=1)

            # Resample if necessary
            if sr != args.target_sr:
                vocal = resample(vocal.astype(float), orig_sr=sr, target_sr=args.target_sr)
                
            # Sample RIRs until RT60 is below max_rt60 and pre_samples are below max_pre_samples

            rir_file = np.random.choice(rir_files)
            rir, sr = read(rir_file, always_2d=True)
            # Take random channel if file is multi-channel
            channel = np.random.randint(0, rir.shape[1])
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
            dry_wet = np.random.uniform(0.1, 1)
            mixture = (1-dry_wet) * vocal + dry_wet * mixture

            # normalize mixture
            loudness_vocal = meter.integrated_loudness(vocal)
            loudness_mixture = meter.integrated_loudness(mixture)
            delta_loudness = loudness_vocal - loudness_mixture
            gain = np.power(10.0, delta_loudness/20.0)
            
            # if gain is inf sample again
            if np.isinf(gain):
                log_skipped_file(id, vocal_file, "Gain is inf or nan", f"{gain}")
                continue
            
            mixture = gain * mixture

            if np.max(np.abs(mixture)) > 1.0:
                mixture = mixture / np.max(np.abs(mixture))

            # Cut long files into pieces
            if len(mixture) >= int((args.cut_length + args.min_length)*args.target_sr):
                long_mixture = mixture
                long_vocal = vocal
                num_splits = int((len(long_mixture) - int(args.min_length*args.target_sr))/int(args.cut_length*args.target_sr)) + 1
                for i in range(num_splits - 1):
                    vocal_start = i*int(args.cut_length*args.target_sr)
                    vocal_end = (i+1)*int(args.cut_length*args.target_sr)
                    mixture = long_mixture[vocal_start:vocal_end]
                    vocal = long_vocal[vocal_start:vocal_end]
                    id = save_files(target_dir, subset, ".", id, vocal_file, vocal_start, vocal_end, rir_file,
                                    channel, gain, rt60, mixture, vocal, args, dry_wet)
                vocal_start = (num_splits - 1)*int(args.cut_length*args.target_sr)
                vocal_end = -1
                mixture = long_mixture[vocal_start:vocal_end]
                vocal = long_vocal[vocal_start:vocal_end]
                id = save_files(target_dir, subset, ".", id, vocal_file, vocal_start, vocal_end, rir_file,
                                channel, gain, rt60, mixture, vocal, args, dry_wet)
            else:
                vocal_start = 0
                vocal_end = -1
                id = save_files(target_dir, subset, ".", id, vocal_file, vocal_start, vocal_end, rir_file,
                                channel, gain, rt60, mixture, vocal, args, dry_wet)

       