from PIL import Image
import numpy as np
import sys
from numpy import linalg as LA
import random


class Img:
    width = 0
    height = 0
    image = Image.new("RGBA", (width, height))
    rays = np.empty(0,)
    spheres = np.empty(0,)
    suns = np.empty(0,)
    expose = -1.0
    origin = np.array([0.0,0.0,0.0])
    forward = np.array([0.0,0.0,-1.0])
    right = np.array([1.0,0.0,0.0])
    up = np.array([0.0,1.0,0.0])
    fisheye = False
    planes = np.empty(0,)
    verts = np.empty(0,)
    aa = -1
    def __init__(self, width,height,image,rays,spheres,suns, expose, up, origin, forward, fisheye, planes, bulbs, aa):
        self.width = width
        self.height = height
        self.image = image
        self.rays = rays
        self.spheres = spheres
        self.suns = suns
        self.expose = expose
        self.fisheye = fisheye
        self.planes = planes
        self.bulbs = bulbs
        self.aa = aa
        #print(up, origin, forward)
        self.origin = origin
        self.forward = forward
        #self.up = up
        temp = np.cross(np.cross(self.forward, up), self.forward)
        self.up = np.cross(self.right, self.forward)
        self.up = self.up / LA.norm(self.up)
        #print(up/ LA.norm(up), self.up)
        self.right = np.cross(self.forward, self.up)
        self.right = self.right / LA.norm(self.right)
        #print(self.forward, self.origin, self.right, self.up)

img = None

class Ray:
    x = 0.0
    y = 0.0
    sx = 0.0
    sy = 0.0
    dir = np.array([0,0,0])
    poi = np.array([0,0,0])
    inormal = np.array([0,0,0])
    t = -1
    hitObject = -1 #remember to change this when triangles come into play (currently keeps index in spheres)
    hitObjectType = 0 #0 for spheres, 1 for planes, etc...
    dontShoot = False
    def __init__(self,sx,sy, x, y):
        self.sx = sx
        self.sy = sy
        self.x = x
        self.y = y
        self.t = -1

    def NormalizeDirection(self):
        #print(forward, right, up)
        dir_raw = 0
        if (img.fisheye):
            if (1 - (self.sx**2) - (self.sy**2) < 0):
                self.dontShoot = True
            dir_raw = np.sqrt(1 - (self.sx**2) - (self.sy**2))*img.forward + self.sx * img.right + self.sy * img.up
        else:
            dir_raw = img.forward + self.sx * img.right + self.sy * img.up
        self.dir = dir_raw / LA.norm(dir_raw)
        #print(img.forward, img.right, img.up)

    def SphereCollision(self, sphere, sphere_idx):
        #sphere is a 4-length array [x,y,z,r]
        c = np.array([float(sphere[0]),float(sphere[1]),float(sphere[2])])
        r = float(sphere[3])
        #print(img.origin, self.dir)
        inside = np.sqrt(np.sum((np.subtract(c, img.origin))**2))**2 < r**2
        t_c = np.dot((c - img.origin), self.dir)
        #print(img.origin)
        if (not inside and t_c < 0):
            return False
        d_squared = np.sum((img.origin + (t_c*self.dir) - c)**2)
        if (not inside and r**2 < d_squared):
            return False
        t_off = np.sqrt(r**2 - d_squared)
        t_temp = 0
        if (inside):
            t_temp = t_c + t_off
        else:
            t_temp = t_c - t_off
        if (t_temp < self.t or self.t == -1):
            self.t = t_temp
            self.poi = self.dir * self.t + img.origin
            self.inormal = ((self.poi - c) / r)
            if (np.dot(self.dir,self.inormal) > 0):
                self.inormal[0] = -self.inormal[0]
                self.inormal[1] = -self.inormal[1]
                self.inormal[2] = -self.inormal[2]
            self.inormal = self.inormal / LA.norm(self.inormal)
            self.hitObject = sphere_idx
        else:
            return False
        self.hitObjectType = 0
        return True
    
    def PlaneCollision(self, plane, plane_idx):
        A = plane[0]
        B = plane[1]
        C = plane[2]
        D = plane[3]
        n = np.array([A,B,C])
        p = (-D*n) / LA.norm(n)
        #print(p)
        t = 0
        if (np.dot(self.dir, n) == 0):
            t = 0
        else:
            t = np.dot((p - img.origin),n) / (np.dot(self.dir, n))
        if (t <= 0):
            return False
        if (t < self.t or self.t == -1):
            self.t = t
            self.poi = self.dir * self.t + img.origin
            self.inormal = n / LA.norm(n)
            self.hitObject = plane_idx
        else:
            return False
        self.hitObjectType = 1
        return True
    
    def CheckShadow():
        return True

class SecondaryRay:
    o = np.array([0,0,0])
    dir = np.array([0,0,0])

    def __init__(self, o, dir):
        self.o = o
        self.dir = dir / LA.norm(dir)

    def ShadowCheck(self):
        #We simply want to check if any spheres lies between the point and the light source
        for i in range(0, len(img.spheres)):
            sphere = img.spheres[i]
            c = np.array([sphere[0],sphere[1],sphere[2]], dtype = float)
            r = float(sphere[3])
            inside = np.sqrt(np.sum((np.subtract(c, img.origin))**2))**2 < r**2
            t_c = np.dot((c - self.o), self.dir)

            if (not inside and t_c < 0):
                continue
            d_squared = np.sum((self.o + (t_c*self.dir) - c)**2)
            if (not inside and r**2 < d_squared):
                continue
            # if (i == 0 or i == 3):
            #     print("shouldn't cast")
            # t_off = np.sqrt(r**2 - d_squared)
            # t_temp = 0
            # if (inside):
            #     t_temp = t_c + t_off
            # else:
            #     t_temp = t_c - t_off
            # # if (t_temp < self.t or self.t == -1):
            # #     self.t = t_temp
            # #     self.poi = self.dir * self.t
            # #     self.inormal = (self.poi - c) / r
            # #     self.hitObject = sphere_idx
            # # else:
            #     return False=
            return True
        return False
    def BulbShadowCheck(self, bulbDist):
        #We simply want to check if any spheres lies between the point and the light source
        for i in range(0, len(img.spheres)):
            sphere = img.spheres[i]
            c = np.array([sphere[0],sphere[1],sphere[2]], dtype = float)
            r = float(sphere[3])
            inside = np.sqrt(np.sum((np.subtract(c, img.origin))**2))**2 < r**2
            t_c = np.dot((c - self.o), self.dir)

            if (not inside and t_c < 0):
                continue
            d_squared = np.sum((self.o + (t_c*self.dir) - c)**2)
            if (not inside and r**2 < d_squared):
                continue
            if (bulbDist < t_c):
                continue
            return True
        return False

        



#Handle ray emission through the 2d grid from the eye.
def Emit():
    img.rays = np.empty((height,width), dtype = Ray)
    #print(len(rays))
    for i in range(0, height):
        row = np.empty((width,), dtype = Ray)
        sy = (height - 2*i) / (max(width, height))
        for j in range(0, width):
            sx = (2*j - width) / (max(width, height))
            curr_ray = Ray(sx,sy, j, i)
            curr_ray.NormalizeDirection()
            #print(curr_ray.dir)
            row[j] = curr_ray
        img.rays[i] = row
    #print(rays)
    RayCollision()


#Read through the existing shapes and determine ray collisions from the emitted rays.
def RayCollision():
    for i in range(0, len(img.rays)):
        for j in range(0,len(img.rays[0])):
            curr_ray = img.rays[i][j]
            if (curr_ray.dontShoot):
                continue
            for k in range(0,len(img.spheres)):
                curr_ray.SphereCollision(img.spheres[k],k)
                # if (curr_ray.hitObject != -1):
                #     print(curr_ray.dir, curr_ray.inormal)
            for l in range(0, len(img.planes)):
                curr_ray.PlaneCollision(img.planes[l], l)
            if (curr_ray.hitObject != -1):
                alpha = 255
                color = None
                if (curr_ray.hitObjectType == 0):
                    # if (spheres[curr_ray.hitObject][7] != None):
                    #     color = TextureMap(curr_ray, spheres[curr_ray.hitObject])
                    # else:
                    if (img.aa == -1):
                        color = CalculateColor(curr_ray, spheres[curr_ray.hitObject], 0)
                    else:
                        runningColor = np.array([0.0,0.0,0.0])
                        num_misses = 0
                        for a in range(0, img.aa):
                            perturb_y = random.uniform(-0.05,0.05)
                            perturb_x = random.uniform(-0.05,0.05)
                            aa_ray_sy = curr_ray.sy + perturb_y
                            aa_ray_sx = curr_ray.sx + perturb_x
                            temp_ray = Ray(aa_ray_sx, aa_ray_sy, curr_ray.x, curr_ray.y)
                            temp_ray.NormalizeDirection()
                            for b in range(0, len(img.spheres)):
                                temp_ray.SphereCollision(img.spheres[b],b)
                            if (temp_ray.hitObject != -1):
                                runningColor += CalculateColor(temp_ray, spheres[temp_ray.hitObject], 0)
                            else:
                                num_misses += 1
                        color = runningColor / img.aa
                        alpha = int((1 - (num_misses / img.aa)) * 255)
                else:
                    color = CalculateColor(curr_ray, planes[curr_ray.hitObject], 1)
                
                image.putpixel((curr_ray.x, curr_ray.y), (int(color[0] * 255),int(color[1] * 255),int(color[2] * 255),alpha))
    image.save(filename)

#returns the color of the pixel
def CalculateColor(ray, obj, hitObjectType):
    lightSum = 0
    if (len(suns) == 0 and len(img.bulbs) == 0):
        return np.array([0,0,0])
    if (hitObjectType == 0):
        sphere_color = np.array([obj[4], obj[5],obj[6]])
        for i in range(0, len(img.suns)):
            sun_color = np.array([suns[i][3],suns[i][4],suns[i][5]])
            sun_pos = np.array([suns[i][0],suns[i][1],suns[i][2]])
            #print(sun_color)
            dir_to_light = sun_pos / LA.norm(sun_pos)
            #print(dir_to_light)
            #Shadow stuff:
            shadowRay = SecondaryRay(ray.poi, dir_to_light)
            isShadow = shadowRay.ShadowCheck()
            shadowVal = 0
            if (isShadow):
                shadowVal = 0
            else:
                shadowVal = 1
            # print(np.dot(ray.inormal, dir_to_light), ray.inormal, dir_to_light)
            lightSum += sphere_color * sun_color * (np.dot(ray.inormal, dir_to_light)) * shadowVal
        for j in range(0, len(img.bulbs)):
            bulb_pos = np.array([bulbs[j][0], bulbs[j][1], bulbs[j][2]])
            bulb_col = np.array([bulbs[j][3], bulbs[j][4], bulbs[j][5]])
            bulb_dir = bulb_pos - ray.poi
            bulb_dir = bulb_dir / LA.norm(bulb_dir)
            bulbRay = SecondaryRay(ray.poi, bulb_dir)
            bulbDist = np.sqrt((bulb_pos[0] - ray.poi[0])**2 + (bulb_pos[1] - ray.poi[1])**2 + (bulb_pos[2] - ray.poi[2])**2)
            bulbIntensity = 1/(bulbDist**2)
            #print(bulbIntensity)
            bulb_dot = np.dot(ray.inormal, bulb_dir)
            if (bulb_dot < 0):
                bulb_dot = 0
            shadowRay = SecondaryRay(ray.poi, bulb_dir)
            isShadow = shadowRay.BulbShadowCheck(bulbDist)
            shadowVal = 0
            if (isShadow):
                shadowVal = 0
            else:
                shadowVal = 1
            lightSum += sphere_color * bulb_col * bulb_dot * bulbIntensity * shadowVal
            #print(sphere_color, bulb_col, bulb_dot, bulbIntensity)
            #print(sphere_color * bulb_col * (np.dot(ray.inormal, bulb_dir)) * bulbIntensity)
    else:
        plane_color = np.array([obj[4],obj[5],obj[6]])
        for i in range(0, len(suns)):
            sun_color = np.array([suns[i][3],suns[i][4],suns[i][5]])
            sun_pos = np.array([suns[i][0],suns[i][1],suns[i][2]])
            dir_to_light = sun_pos / LA.norm(sun_pos)
            #print(dir_to_light)
            #Shadow stuff:
            shadowRay = SecondaryRay(ray.poi, dir_to_light)
            isShadow = shadowRay.ShadowCheck()
            shadowVal = 0
            if (isShadow):
                shadowVal = 0
            else:
                shadowVal = 1
            lightSum += plane_color * sun_color * (np.dot(ray.inormal, dir_to_light)) * shadowVal 
    #print(lightSum)
    for j in range(0, 3):
    # print(":LKDSJ:LKJ")
        if (expose != -1.0):
            lightSum[j] = 1 - np.exp(-(expose * lightSum[j]))
        if (lightSum[j] > 1):
            lightSum[j] = 1
        elif (lightSum[j] < 0):
            lightSum[j] = 0
        #print(lightSum[j])
        #exposure calculation:
        
        #print(lightSum[j])
        if (lightSum[j] <= 0.0031308):
            lightSum[j] *= 12.92
        else:
            lightSum[j] = (1.055*(lightSum[j]**(1/2.4))) - 0.055
    # print(lightSum[j])
    
    #print("AFTER: " + str(lightSum))
    #print(lightSum)
    return lightSum

with open(sys.argv[1], 'r') as f:
    
    #print(f)
    lines = f.readlines()
    #file setup state
    png_filename = ""
    
    #image = Image.new("RGBA", (width, height))
    #mode state (Ignore until done with core functionality)
    depth = False #depth buffer and depth tests
    sRGB = False   # sRGB conversions of color before saving to png file
    hyp = False # enables hyperbolic interpolation of depth, color, and texture coords
    fsaa = 0 #full screen anti-aliasing/multisampling level ranges from 1 to 8
    cull = False #enables back-face culling
    decals = False #when drawing transparent textures, include the vertex colors underneath
    frustum = False #enables frustum clipping
    image = Image.new("RGBA", (0,0))
    #uniform state
    texture = "" #will be a filename
    uniformMatrix = [] #Will be a 4x4 matrix n0-n15 multiplied by (x,y,z,w)

    #buffer provision
    positions_raw = [] #will contain a number first representing size which will be used to group together following listed numbers
    #Note x= -1 to 1 is left to right, y = -1 to 1 is top to bottom
    positions = [] #positions after viewport transformations are applied
    color = [1,1,1] #will contain a size value used like position of 3 or 4. 3 for RGB, 4 for RGBA. A is alpha for transparency
    textcoord = [] #size is always 2. gives texel coordinates of size 2.
    point_size = [] #size is always 1. (size of rendered points)
    elements = [] #all indices are non-negative integers

    #RAYTRACER STUFF:
    spheres = [] #contains spheres as 4len arrays
    suns = [] #contains suns as 3len arrays
    expose = -1.0
    up = np.array([0.0,1.0,0.0])
    origin = np.array([0.0,0.0,0.0])
    forward = np.array([0.0,0.0,-1.0])
    fisheye = False
    panorama = False
    planes = [] #contains planes as a 7 len array with ABCD and rgb
    bulbs = []
    aa_rays = -1
    #lines in a file will either manipulate state or draw current state
    for line in lines:
        line = line.strip()
        #print(line)
        curr_str = ""
        item_idx = 0 #this pertains to each item of interest in the string
        char_idx = 0 #this pertains to the char index in the line
        for c in line:
            if (c != " "):
                curr_str += c
            #=========================================PNG==================================================
            if (curr_str == "png"):
                #print(curr_str)
                curr_str = ""
                item_idx += 1
                char_idx += 1
                curr_substr = ""
                for d in range(char_idx, len(line)):
                    if (line[d] == " "):
                        if (len(curr_substr) != 0):
                            if (item_idx == 1):
                                width = int(curr_substr)
                            elif(item_idx == 2):
                                height = int(curr_substr)
                            item_idx += 1
                            curr_substr = ""
                    elif (d >= len(line) - 1):
                        filename = curr_substr 
                        filename += line[len(line) - 1]
                    else:
                        curr_substr += line[d]
                #print("Finished processing png file information. width is " + str(width) + " height is " + str(height) + " filename is " + str(filename))
                image = Image.new("RGBA", (width, height))
                break #break here because we finished processing a line
            #==========================================PNG==============================================

            #==========================================SPHERE===========================================
            if (curr_str == "sphere"):
                curr_str = ""
                item_idx += 1
                char_idx += 1
                curr_substr = ""
                x = 0
                y = 0
                z = 0 
                r = 0
                for d in range(char_idx, len(line)):
                    if (line[d] == " "):
                        if (len(curr_substr) != 0):
                            if (item_idx == 1):
                                x = float(curr_substr)
                            elif(item_idx == 2):
                                y = float(curr_substr)
                            elif(item_idx == 3):
                                z = float(curr_substr)
                            item_idx += 1
                            curr_substr = ""
                    elif (d >= len(line) - 1):
                        r = curr_substr 
                        r += line[len(line) - 1]
                    else:
                        curr_substr += line[d]
                #print(x, y, z, r)                
                sphere = np.array([[float(x),float(y),float(z),float(r),float(color[0]),float(color[1]),float(color[2])]])
                if (len(spheres) == 0):
                    spheres = sphere
                else:
                    spheres = np.concatenate((spheres, sphere))

                break
            #=========================================SPHERES========================================

            #=========================================SUNS===========================================
            if (curr_str == "sun"):
                curr_str = ""
                item_idx += 1
                char_idx += 1
                curr_substr = ""
                x = 0
                y = 0
                z = 0 
                for d in range(char_idx, len(line)):
                    if (line[d] == " "):
                        if (len(curr_substr) != 0):
                            if (item_idx == 1):
                                x = float(curr_substr)
                            elif(item_idx == 2):
                                y = float(curr_substr)
                            item_idx += 1
                            curr_substr = ""
                    elif (d >= len(line) - 1):
                        z = curr_substr 
                        z += line[len(line) - 1]
                    else:
                        curr_substr += line[d]
                #print(x, y, z, r)                
                sun = np.array([[float(x),float(y),float(z),float(color[0]),float(color[1]),float(color[2])]])
                if (len(suns) == 0):
                    suns = sun
                else:
                    suns = np.concatenate((suns, sun))

                break
            #==========================================SUN================================================

            #===========================================COLOR=============================================
            if (curr_str == "color"):
                curr_str = ""
                item_idx += 1
                char_idx += 1
                curr_substr = ""
                for d in range(char_idx, len(line)):
                    if (line[d] == " "):
                        if (len(curr_substr) != 0):
                            if (item_idx == 1):
                                color[0] = float(curr_substr)
                            elif(item_idx == 2):
                                color[1] = float(curr_substr)
                            item_idx += 1
                            curr_substr = ""
                    elif (d >= len(line) - 1):
                        color[2] = curr_substr 
                        color[2] += line[len(line) - 1]
                    else:
                        curr_substr += line[d]
                break
            #======================================COLOR===============================================

            #======================================EXPOSE================================================
            if (curr_str == "expose"):
                curr_str = ""
                item_idx += 1
                char_idx += 1
                curr_substr = ""
                for d in range(char_idx, len(line)):
                    if (d >= len(line) - 1):
                        expose = curr_substr 
                        expose += line[len(line) - 1]
                    else:
                        curr_substr += line[d]
                expose = float(expose)
                break
            #======================================EXPOSE==============================================

            #======================================UP==================================================
            if (curr_str == "up"):
                curr_str = ""
                item_idx += 1
                char_idx += 1
                curr_substr = ""
                #print(line)
               # print(len(line))
                for d in range(char_idx, len(line)):
                    #print(line[d])
                    if (line[d] == " "):
                        if (len(curr_substr) != 0):
                            if (item_idx == 1):
                                up[0] = float(curr_substr)
                            elif(item_idx == 2):
                                up[1] = float(curr_substr)
                            item_idx += 1
                            #print(curr_substr)
                            curr_substr = ""
                    elif (d >= len(line) - 1):
                        temp = curr_substr 
                        temp += line[len(line) - 1]
                        up[2] = temp
                    else:
                        curr_substr += line[d]
                        #print("Else" + str(curr_substr))
                break
            #===================================UP=====================================================

            #===================================EYE(origin)============================================
            if (curr_str == "eye"):
                curr_str = ""
                item_idx += 1
                char_idx += 1
                curr_substr = ""
                for d in range(char_idx, len(line)):
                    if (line[d] == " "):
                        if (len(curr_substr) != 0):
                            if (item_idx == 1):
                                origin[0] = float(curr_substr)
                            elif(item_idx == 2):
                                origin[1] = float(curr_substr)
                            item_idx += 1
                            curr_substr = ""
                    elif (d >= len(line) - 1):
                        temp = curr_substr 
                        temp += line[len(line) - 1]
                        origin[2] = temp
                    else:
                        curr_substr += line[d]
                break
            
            #===================================EYE(origin)===========================================

            #===================================FORWARD===============================================
            if (curr_str == "forward"):
                curr_str = ""
                item_idx += 1
                char_idx += 1
                curr_substr = ""
                for d in range(char_idx, len(line)):
                    if (line[d] == " "):
                        if (len(curr_substr) != 0):
                            if (item_idx == 1):
                                forward[0] = float(curr_substr)
                            elif(item_idx == 2):
                                forward[1] = float(curr_substr)
                            item_idx += 1
                            curr_substr = ""
                    elif (d >= len(line) - 1):
                        temp = curr_substr 
                        temp += line[len(line) - 1]
                        forward[2] = temp
                    else:
                        curr_substr += line[d]
                break
            #==================================FISHEYE================================================
            if (curr_str == "fisheye"):
                fisheye = True
                break
            #===================================FISHEYE===============================================

            #===================================PANORAMA==============================================
            #I didn't do panorama just fisheye
            if (curr_str == "panorama"):
                panorama = True
                break
            #===================================PANORAMA=================================================

            #==================================PLANE=====================================================
            if (curr_str == "plane"):
                curr_str = ""
                item_idx += 1
                char_idx += 1
                curr_substr = ""
                A = 0
                B = 0
                C = 0 
                D = 0
                for d in range(char_idx, len(line)):
                    if (line[d] == " "):
                        if (len(curr_substr) != 0):
                            if (item_idx == 1):
                                A = float(curr_substr)
                            elif(item_idx == 2):
                                B = float(curr_substr)
                            elif(item_idx == 3):
                                C = float(curr_substr)
                            item_idx += 1
                            curr_substr = ""
                    elif (d >= len(line) - 1):
                        D = curr_substr 
                        D += line[len(line) - 1]
                    else:
                        curr_substr += line[d]
                #print(x, y, z, r)                
                plane = np.array([[float(A),float(B),float(C),float(D),float(color[0]),float(color[1]),float(color[2])]])
                if (len(planes) == 0):
                    planes = plane
                else:
                    planes = np.concatenate((planes, plane))

                break
            #=============================================PLANE=======================================================

            #==============================================image======================================================
            # if (curr_str == "texture"):
            #     curr_str = ""
            #     item_idx += 1
            #     char_idx += 1
            #     curr_substr = ""
            #     for d in range(char_idx, len(line)):
            #         if (d >= len(line) - 1):
            #             image_file = curr_substr 
            #             image_file += line[len(line) - 1]
            #             print(image_file)
            #             if (image_file == " none"):
            #                 image_file = None
            #         else:
            #             curr_substr += line[d]
            #     break

            #==========================================image======================================================

            #===========================================BULB=======================================================
            if (curr_str == "bulb"):
                curr_str = ""
                item_idx += 1
                char_idx += 1
                curr_substr = ""
                x = 0
                y = 0
                z = 0 
                for d in range(char_idx, len(line)):
                    if (line[d] == " "):
                        if (len(curr_substr) != 0):
                            if (item_idx == 1):
                                x = float(curr_substr)
                            elif(item_idx == 2):
                                y = float(curr_substr)
                            item_idx += 1
                            curr_substr = ""
                    elif (d >= len(line) - 1):
                        z = curr_substr 
                        z += line[len(line) - 1]
                    else:
                        curr_substr += line[d]
                #print(x, y, z, r)                
                bulb = np.array([[float(x),float(y),float(z),float(color[0]),float(color[1]),float(color[2])]])
                if (len(bulbs) == 0):
                    bulbs = bulb
                else:
                    bulbs = np.concatenate((bulbs, bulb))

                break

            
        #====================================================BULB=================================

        #====================================================aa===================================
            if (len(suns) > 0 and curr_str == "aa"):
                curr_str = ""
                item_idx += 1
                char_idx += 1
                curr_substr = ""
                for d in range(char_idx, len(line)):
                    if (d >= len(line) - 1):
                        aa_rays = curr_substr 
                        aa_rays += line[len(line) - 1]
                        aa_rays = int(aa_rays)
                    else:
                        curr_substr += line[d]
                break


            char_idx += 1
    img = Img(width, height, image, np.empty((0,)), spheres, suns, expose, up, origin, forward, fisheye, planes, bulbs, aa_rays)

    Emit()