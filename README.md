# K means clustering for images 

# Usage
```bash
  cargo run <path to image> <k>
```

# Example:

| Input Image | Output (k = 50) |
| ------------| ------ |
| ![image](https://github.com/user-attachments/assets/2dc60967-7b49-4769-9431-f545aa082728) | ![image](https://github.com/user-attachments/assets/3ce68439-ca65-44fe-a7e7-024f98413160)


# Limitations
- Arbitrary n-dimensional data isn't supported and isn't planned to be supported. This is only really meant for image segmentation
- No convergence tracking yet, currently the number of k-means iterations is hardcoded
- K (#clusters) must be specified manually at the moment

# Planned
- Add support for reading-out results from the GPU. Currently the results can only be viewed visually
- Auto-select k instead of specifying it manually, using the [Elbow Method](https://en.wikipedia.org/wiki/Elbow_method_(clustering)) or similar
- Convergence tracking
