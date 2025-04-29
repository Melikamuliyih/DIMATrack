# DIMATrack

#### Our DIMATrack method's overall architecture
![DIMATrack](https://github.com/user-attachments/files/18613889/DIMATrack1.pdf)


> [**Liu, S., Sinishaw, M.L., Zheng, L. (2025). DIMATrack: Dimension Aware Data Association for Multi-Object Tracking**]([https://www.sciencedirect.com/science/article/pii/S0141938224000465](https://doi.org/10.1007/978-981-96-5815-2_2), 

#Abstract
Multi-Object Tracking (MOT) is crucial for real-world ap-
plications like video surveillance, where it aims to detect and maintain
consistent identifiers for objects across video frames. However, MOT
methods often struggle with objects that are heavily overlapped due
to occlusion or exhibit diverse poses due to non-linear motion. In this
paper, we propose a robust tracking-by-detection method named DIMA-
Track. It incorporates the Kalman Filter for precise trajectory prediction,
and a novel Dimension Aware Intersection-over-Union (DIMA-IoU) met-
ric for enhanced data association. DIMA-IoU improves upon standard
IoU by integrating both height-aware and width-aware measurements,
improving association accuracy in complex scenarios and during occlu-
sions. By integrating these components, DIMATrack effectively leverages
weak cues that are often overlooked by conventional methods, which rely
on appearance or spatial information. Extensive experiments on three
benchmarks demonstrate the superior performance of our DIMATrack,
particularly in challenging tracking environments. The code is available
at https://github.com/Melikamuliyih/DIMATrack.





## Citation
```
Liu, S., Sinishaw, M.L., Zheng, L. (2025). DIMATrack: Dimension Aware Data Association for Multi-Object Tracking. In: Didyk, P., Hou, J. (eds) Computational Visual Media. CVM 2025. Lecture Notes in Computer Science, vol 15665. Springer, Singapore. https://doi.org/10.1007/978-981-96-5815-2_2
```
