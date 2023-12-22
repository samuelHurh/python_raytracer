# python_raytracer
A program that creates raytracing generated images from text input that specifies aspects like light sources and geometry

To run, first pull onto your machine and navigate via the terminal into the directory you pulled into.
Using the makefile type:
make run file = "filename.txt"
which will run the "filename" you specify and generate its corresponding image (delete and run in order to verify program generates the right image)

Some examples of what can be created:
![alt_text](https://github.com/samuelHurh/python_raytracer/blob/main/raytracing/aa.png?raw=true)
Anti-Aliasing
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/raytracing/behind.png?raw=true)
A sphere placed behind the camera
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/raytracing/bulb.png?raw=true)
Raytracing example with a point light source
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/raytracing/expose1.png?raw=true)
Raytracing where the expose image effect is decreased
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/raytracing/expose2.png?raw=true)
Raytracing where the expose image effect is increased
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/raytracing/fisheye.png?raw=true)
Raytracing through a fisheye lens
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/raytracing/overlap.png?raw=true)
Raytracing testing to see that the z-index of overlapping geometry is properly handled
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/raytracing/plane.png?raw=true)
Raytracing with a plane geometric object
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/raytracing/shadow-basic.png?raw=true)
Raytracing with basic shadows
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/raytracing/shadow-bulb.png?raw=true)
Raytracing with shadows and a point light source
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/raytracing/shadow-suns.png?raw=true)
Raytracing with shadows and different colored suns
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/raytracing/sphere.png?raw=true)
Raytracing with basic sphere geometric objects
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/raytracing/sun.png?raw=true)
Raytracing with a sun (unlimited distance directional light source)
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/raytracing/suns.png?raw=true)
Raytracing with multiple colorful suns
![alt text](https://github.com/samuelHurh/python_raytracer/blob/main/raytracing/view.png?raw=true)
Raytracing on a scene with a differed camera perspective



