# CloudNet weakly labeled project
## Abstract 
Semantic segmentation is the task of labeling each pixel in an image with the class of which the pixel is a part of. In order to train a deep convolutional network that will accomplish this task, it requires a lot of hand-drawn segmantation maps and therefore a lot of resources.

We wanted to check whether it is possible to save resources on the manual labeling of cloud segmentation maps and still achieve competitive results.

In order to find out, we trained a deep convolutional network for semantic segmentation of cloud images on both fully-labeled data (full segmentation maps) and weakly-labeled data (scribbles) and compared the results achieved by both methods.

## Examples
### Weakly labeled (Scribbles) example:
![Scribbles example](https://github.com/avivcaspi/Cloudnet_project/blob/master/report/scribbles%20segmentation%20example.jpg)

![Cloud scribbles example](https://github.com/avivcaspi/Cloudnet_project/blob/master/report/scribbles_example.png)

### Cloud Net Architecture:
![Cloud net](https://github.com/avivcaspi/Cloudnet_project/blob/master/report/cloudnet%2B%20architecture.png)

### Results using scribbles as labels:
![Results](https://github.com/avivcaspi/Cloudnet_project/blob/master/report/weakly%20training%20epoch%2050%20only%20celoss(6%20images).png)

### Comparison between all models:
![Results comparison](https://github.com/avivcaspi/Cloudnet_project/blob/master/report/inference%20networks%20comparison.png)
