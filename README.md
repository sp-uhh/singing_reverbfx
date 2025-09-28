# SingingReverbFX

This repository provides generation scripts for the SingingReverbFX benchmark — a dataset designed to evaluate dereverberation methods for singing voices processed with various reverb effects.

## Overview

The SingingReverbFX benchmark simulates reverberant singing voices by convolving clean vocal recordings with ReverbFX room impulse responses (RIRs). This setup enables testing and development of dereverberation algorithms specifically for singing voice signals.

You can download the ReverbFX RIRs [here](https://zenodo.org/records/16186381).

## Included Datasets

The following singing voice datasets are used:

- [OpenSinger](https://drive.google.com/file/d/1EofoZxvalgMjZqzUEuEdleHIZ6SHtNuK/view)  
- [M4Singer](https://drive.google.com/file/d/1xC37E59EWRRFFLdG3aJkVqwtLDgtFNqW/view)  
- [CSD](https://zenodo.org/records/4785016/files/CSD.zip)  
- [PJS](https://drive.google.com/file/d/1hPHwOkSe2Vnq6hXrhVtzNskJjVMQmvN_/view)  
- [Opencpop](https://wenet.org.cn/opencpop/download/)  
- [NUS-48E](https://drive.google.com/drive/folders/12pP9uUl0HTVANU3IPLnumTJiRjPtVUMx)  
- [NHSS](https://hltnus.github.io/NHSSDatabase/)  


## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/sp-uhh/singing_reverbfx.git
cd singing_reverbfx
pip install -r requirements.txt
```

## Usage

1. Download All Datasets

Download each of the listed singing voice datasets and organize them in a root directory of your choice.

2. Create the Test Set

Run the following command to generate the test set:

```bash
python create_test.py --data_dir <ROOT_DIR>
```

Replace <ROOT_DIR> with the path to the directory containing the downloaded datasets and where SingingReverbFX will be created.

3. Create Training and Validation Sets

Run:

```bash
python create_train_valid.py --data_dir <ROOT_DIR>
```

Replace <ROOT_DIR> with the same root directory as above.

## Citation

If you use ReverbFX or SingingReverbFX in your research, please cite our paper

```bibtex
@inproceedings{richter2025reverbfx,
    title={{ReverbFX}: A Dataset of Room Impulse Responses Derived from Reverb Effect Plugins for Singing Voice Dereverberation},
    author={Richter, Julius and Svajda, Till and Gerkmann, Timo},
    booktitle={ITG Conference on Speech Communication},
    year={2025}
}
```

## License

We obtained explicit permission from the plugin developers to generate and publish RIRs derived from the plugins for
non-commercial research purposes. The released dataset does not contain any part of the original software. ReverbFXis released under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). It may be used, shared, and adapted for non-commercial research and educational purposes only. Commercial use is strictly prohibited. Proper attribution must be given to the dataset authors and the original plugin developers.