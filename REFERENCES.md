# References

fastpose is a numba port of algorithms implemented in
[PoseLib](https://github.com/PoseLib/PoseLib). If you use fastpose, please
consider citing PoseLib and the original papers behind the specific solver(s) you
use, listed below by backend (see [README.md](README.md#backend-notes) for
which file implements what).

## PoseLib

```bibtex
@misc{poselib,
  title  = {{PoseLib -- Minimal Solvers for Camera Pose Estimation}},
  author = {Viktor Larsson and contributors},
  url    = {https://github.com/PoseLib/PoseLib},
  year   = {2020}
}
```

## Fundamental matrix (7-point)

Hartley-style isotropic point normalization used for numerical conditioning:

```bibtex
@article{hartley1997defense,
  title   = {In defense of the eight-point algorithm},
  author  = {Hartley, Richard I.},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {19},
  number  = {6},
  pages   = {580--593},
  year    = {1997},
  publisher = {IEEE}
}
```

## Calibrated relative pose (5-point)

`src/fastpose/solvers/essential.py` follows the nullspace + action-matrix
formulation of:

```bibtex
@article{stewenius2006recent,
  title   = {Recent developments on direct relative orientation},
  author  = {Stew{\'e}nius, Henrik and Engels, Christopher and Nist{\'e}r, David},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume  = {60},
  number  = {4},
  pages   = {284--294},
  year    = {2006},
  publisher = {Elsevier}
}
```

building on the original 5-point algorithm:

```bibtex
@article{nister2004efficient,
  title   = {An efficient solution to the five-point relative pose problem},
  author  = {Nist{\'e}r, David},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {26},
  number  = {6},
  pages   = {756--770},
  year    = {2004},
  publisher = {IEEE}
}
```

## Varying-focal relative pose

The overall pipeline was not published as standalone work, but it was used as a baselin in the following paper:

```bibtex
@article{kocur2026minimal,
  title={Are Minimal Radial Distortion Solvers Really Necessary for Relative Pose Estimation? V. Kocur et al.},
  author={Kocur, Viktor and Tzamos, Charalambos and Ding, Yaqing and Haladova, Zuzana Berger and Sattler, Torsten and Kukelova, Zuzana},
  journal={International Journal of Computer Vision},
  volume={134},
  number={2},
  pages={48},
  year={2026},
  publisher={Springer}
}
```

The Bougnoux formula used to recover per-image focal lengths from the
7-point fundamental matrix used for all solutions:

```bibtex
@inproceedings{bougnoux1998projective,
  title     = {From projective to Euclidean space under any practical situation, a criticism of self-calibration},
  author    = {Bougnoux, Sylvain},
  booktitle = {Proceedings of the Sixth International Conference on Computer Vision (ICCV)},
  pages     = {790--796},
  year      = {1998}
}
```

## Shared-focal relative pose (6-point)

`src/fastpose/solvers/shared_focal.py` is a port of PoseLib's
`relpose_6pt_focal.cc`, based on:

```bibtex
@article{stewenius2008minimal,
  title   = {A minimal solution for relative pose with unknown focal length},
  author  = {Stew{\'e}nius, Henrik and Nist{\'e}r, David and Kahl, Fredrik and Schaffalitzky, Frederik},
  journal = {Image and Vision Computing},
  year    = {2008},
  publisher = {Elsevier}
}
```



The original PoseLib solver was generated using:
```bibtex
@inproceedings{larsson2017efficient,
  title={Efficient solvers for minimal problems by syzygy-based reduction},
  author={Larsson, Viktor and Astrom, Kalle and Oskarsson, Magnus},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={820--829},
  year={2017}
}
```


## Absolute pose (P3P)

`src/fastpose/solvers/p3p.py` is a port of PoseLib's `p3p.cc`:

```bibtex
@inproceedings{ding2023revisiting,
  title     = {Revisiting the P3P problem},
  author    = {Ding, Yaqing and Yang, Jian and Larsson, Viktor and Olsson, Carl and {\AA}str{\"o}m, Kalle},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023}
}
```

## Absolute pose with unknown focal (P4Pf)

`src/fastpose/solvers/p4pf.py` is a port of PoseLib's `p4pf.cc` and
`misc/re3q3.cc`; the 3Q3 (three-quadrics) system solver is from:

```bibtex
@inproceedings{kukelova2016efficient,
  title     = {Efficient intersection of three quadrics and applications in computer vision},
  author    = {Kukelova, Zuzana and Heller, Jan and Fitzgibbon, Andrew},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016}
}
```

## Monodepth-assisted relative pose

`src/fastpose/solvers/monodepth.py` and `src/fastpose/refiners/monodepth.py`
port PoseLib's `relpose_monodepth_3pt*.cc` solvers and refiners, introduced
in:

```bibtex
@inproceedings{ding2025reposed,
  title     = {{RePoseD}: Efficient Relative Pose Estimation With Known Depth Information},
  author    = {Ding, Yaqing and Kocur, Viktor and V{\'a}vra, V{\'a}clav and Berger Haladov{\'a}, Zuzana and Yang, Jian and Sattler, Torsten and Kukelova, Zuzana},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```
