# bee-sperma-analysis
Automation of the analysis of bee's sperma by computer vision. Detection and tracking the cells from videos obtained by a microscope. 

## Goals

- Performance a detection of the different cells, obtaining the position and the form of each cell
  - Via classic vision computation
  - Via machine learning
- Track each cell from video using filters, so it is possible to define *good* from bad *cells* 

## Author

Pablo Luesia Lahoz
> 698387@unizar.es

### Detection

Using **OpenCV** with **Python**, a classic computer vision aproximation to detect the different cells it is being implementing.

So far, a **Canny edge detector** is being used to codify each cell. The edges will be represented by interest points, and the inner points of the cell will be got from those. Each point will have an angle direction, so it is possible to obtain the most adecuate point from the same cell. 
