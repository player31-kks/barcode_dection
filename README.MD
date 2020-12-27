# barcode detection

1. sobel filter
2. Average Filter
3. Threshold
4. opending & closing
5. contour

## sobel filter
sobel filter is used in edge detection algorithms where it creates an image emphasising edges. i used sobel filter that find x's axis and y's axis edge

<center><img src="https://user-images.githubusercontent.com/57718605/103171506-3bc21e00-4890-11eb-86e2-5f941a673dd1.PNG" width="400" height="300"></center>

## Average Filter
average filter is a simple, intuitive and easy to implement method of smoothing images and remove noise

<center><img src="https://user-images.githubusercontent.com/57718605/103172023-d07a4b00-4893-11eb-8224-2f64d22779ea.PNG" width="200" height="250"></center>

## Threshold
For every pixel, the same threshold value is applied If the pixel value is smaller than the threshold, it is set to 0, otherwise it is set to a maximum value(255).

<center><img src="https://user-images.githubusercontent.com/57718605/103172157-d6246080-4894-11eb-890d-5ad663c77235.png" width="200" height="250"></center>

## opening and closing
Opening removes small objects from the foreground(bright) of an image very simply, an opening is defined as an erosion followed by a dilation using same structing element for both operation. while closing removes small holes in the foreground,changing small islands of background into foreground. in the same way with opeing, it defined as an dilation followed by erosion using same structing element

<center><img src="https://user-images.githubusercontent.com/57718605/103172504-5cda3d00-4897-11eb-9522-1148989c7009.PNG" width="500" height="400"></center>

## contour(result)
<center><img src="https://user-images.githubusercontent.com/57718605/103172594-df62fc80-4897-11eb-96a2-1c67fc0a4e57.png" width="500" height="400"></center>
