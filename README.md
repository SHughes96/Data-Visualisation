# Data-Visualisation

This repository is designed to highlight examples of the data visulisations that I have created over the course of my career. Many of these examples are from my PhD thesis and includes:
- Sigma clipping of data extracted from my calibration database
- Least squares fitting over a 2D to create surface maps
- Spline fitting of data and residual plotting
- Vector diagrams displaying instrument concepts

There is also a folder containing posters I created and presented at the SPIE astronomical telescopes and instrumentation 2022 conference.

---

## Sigma Clipping

The plot below displays the corrective movement data recorded for a single optical fibre being placed multiple times by a positioning robot. Each row represents a type of movement, either movement from one coordinate to another, a movement from a coordinate to it's park position, or a movement from it's park position to a coordinate.

From left to right the panels represent the number of sigma clipping iterations applied to the display its impact on the number of data points excluded by the process, as well as the mean offset.

This plot was designed to investigate the filtering level needed to new data to prevent extreme values skewing the corrective offset applied to each movement over time.
![alt text](data_vis_examples/Fibre_192_sigma_clipping.png)


---
## Least Squares Fitting
