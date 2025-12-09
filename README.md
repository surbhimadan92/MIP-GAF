# [MIP-GAF: A MLLM-annotated Benchmark for Most Important Person Localization and Group Context Understanding](https://ieeexplore.ieee.org/abstract/document/10943490/)
--- 

This is an official repository for the paper: "MIP-GAF: A MLLM-annotated Benchmark for Most Important Person Localization and Group Context Understanding", WACV 2025.


## Environment
We recommend running the code using <b>Pytorch 1.13.1</b> or higher version.
 ```bash
conda env create -f environment.yml
``` 

## Training

Download stage 2 weights from [TRIS](https://github.com/fawnliu/TRIS) Repository

Train using the command:
```bash
python train_mip.py
``` 

Inference links and checkpoints will be shared soon.
<!--https://drive.google.com/file/d/1V9JRRli1D_sYKnoNDRPpPgNx7E2K4W1Q/view?usp=sharing-->

## Citation
If you find this dataset helpful, please cite:
```bibtex
@inproceedings{madan2025mip,
  title={MIP-GAF: A MLLM-annotated Benchmark for Most Important Person Localization and Group Context Understanding},
  author={Madan, Surbhi and Ghosh, Shreya and Sookha, Lownish Rai and Ganaie, MA and Subramanian, Ramanathan and Dhall, Abhinav and Gedeon, Tom},
  booktitle={2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={1467--1476},
  year={2025},
  organization={IEEE}
}
```

For any enquiry please write to "surbhi.19csz0011@iitrpr.ac.in"

## Acknowledgement
Much of this code has been borrowed from [TRIS](https://github.com/fawnliu/TRIS)
