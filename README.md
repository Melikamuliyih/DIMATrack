# DIMATrack

#### Our DIMATrack method's overall architecture
[DIMATrack1.pdf](https://github.com/user-attachments/files/19950743/DIMATrack1.pdf)
![image](https://github.com/user-attachments/assets/1c67ae4d-7058-40f4-a707-8524717e71c8)


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

## Tracking performance
### Results on MOT Datasets
| Dataset    |  MOTA | IDF1  | FP | FN | FPS |
|--------------|-----------|-------|----------|----------|--------|
|MOT17       | 80.7% | 79.0% | 2572 | 7398| 31.2 |
|MOT20       | 77.1% | 78.2% |49615 | 90245 | 15.3 |

### Results on DanceTrack Dataset
| Dataset    |  MOTA | IDF1  | DetA |AssA |
|--------------|-----------|-------|----------|----------|
|DanceTrack  | 92.5% | 79.9% |84.4% | 65.5 |



## Citation
```
Liu, S., Sinishaw, M.L., Zheng, L. (2025). DIMATrack: Dimension Aware Data Association for Multi-Object Tracking. In: Didyk, P., Hou, J. (eds) Computational Visual Media. CVM 2025. Lecture Notes in Computer Science, vol 15665. Springer, Singapore. https://doi.org/10.1007/978-981-96-5815-2_2
```
