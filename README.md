# python_raytracer
A program that creates raytracing generated images from text input that specifies aspects like light sources and geometry

To run, first pull onto your machine and navigate via the terminal into the directory you pulled into.
Using the makefile type:
make run file = "filename.txt"
which will run the "filename" you specify and generate its corresponding image (delete and run in order to verify program generates the right image)

Some examples of what can be created:
!(https://github.com/samuelHurh/python_raytracer/blob/main/rasterizer/aa.png?raw=true)
Anti-Aliasing
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/rasterizer/behind.png?raw=true)
A sphere placed behind the camera
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/rasterizer/bulb.png?raw=true)
Raytracing example with a point light source
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/rasterizer/expose1.png?raw=true)
Raytracing where the expose image effect is decreased
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/rasterizer/expose2.png?raw=true)
Raytracing where the expose image effect is increased
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/rasterizer/fisheye.png?raw=true)
Raytracing through a fisheye lens
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/rasterizer/overlap.png?raw=true)
Raytracing testing to see that the z-index of overlapping geometry is properly handled
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/rasterizer/plane.png?raw=true)
Raytracing with a plane geometric object
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/rasterizer/shadow-basic.png?raw=true)
Raytracing with basic shadows
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/rasterizer/shadow-bulb.png?raw=true)
Raytracing with shadows and a point light source
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/rasterizer/shadow-suns.png?raw=true)
Raytracing with shadows and different colored suns
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/rasterizer/sphere.png?raw=true)
Raytracing with basic sphere geometric objects
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/rasterizer/sun.png?raw=true)
Raytracing with a sun (unlimited distance directional light source)
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/rasterizer/suns.png?raw=true)
Raytracing with multiple colorful suns
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/rasterizer/view.png?raw=true)
Raytracing on a scene with a differed camera perspective



