# Data-Visualisation

This repository is designed to highlight examples of the data visualisations that I have created throughout my career. Many of these examples are from my PhD thesis and include:
- Sigma clipping of data extracted from my calibration database
- Least squares fitting over a 2D to create surface maps
- Spline fitting of data and residual plotting
- Vector diagrams displaying instrument concepts

Each of these examples has been created with Python using packages such as Matplotlib, Seaborn, and Scikit-learn amongst others.

There is also a folder containing posters I created and presented at the SPIE Astronomical Telescopes and Instrumentation 2022 conference.

---

## Sigma Clipping

The plot below displays the corrective movement data recorded for a single optical fibre being placed multiple times by a positioning robot. Each row represents a type of movement, either a movement from one coordinate to another, a movement from a coordinate to its park position, or a movement from its park position to a coordinate.

From left to right the panels represent the number of sigma clipping iterations applied to the display its impact on the number of data points excluded by the process, as well as the mean offset.

This plot was designed to investigate the filtering level needed for new data to prevent extreme values from skewing the corrective offset applied to each movement over time.
![alt Sigma Clipping](data_vis_examples/Fibre_192_sigma_clipping.png)


---
## Least Squares Fitting

I created the plot below to create a map of the field plate height where optical fibres were being placed. Each map is unique to the positioning robot used to measure the height. These maps were produced by completing a least squares fit across a regular grid of points after a series of height measurements were taken using each positioning robot. The left and right-hand sides represent one of the two robots.

![alt 2D profile maps](data_vis_examples/Nona_Morta_A_combined_ZD40.png)
