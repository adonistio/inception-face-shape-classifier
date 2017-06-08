#This script contains a couple of image pre-processing and augmentation functions like squaring an image, filters, blurs, zoom, rotate, flip, and recolor

import subprocess
from PIL import Image
from PIL import ImageFilter, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import pathlib
from datetime import datetime
import time
import tensorflow as tf, sys
import os
import numpy as np
import cv2

import os

def plot_images(image, Caption1):

    plt.close()
    
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'Arial'
    
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)
    xlabel = Caption1
    ax.set_xlabel(xlabel)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    
    #plt.close()

def make_img_square(imdir,outdir):
#NOTE: Error if picture has black borders :( because im.getbbox ignores black pixels
    im = Image.open(imdir)
    width, height = im.size
    print(width, height)
    if (width != height):
        size = max(width, height)
        a4im = Image.new('RGB',(size, size), (255, 255, 255)) #second argument is size, 3rd is color padded
        a4im.paste(im, im.getbbox())  # Not centered, top-left corner
        a4im.save(outdir, 'JPEG', quality=100)
    else: im.save(outdir, 'JPEG', quality=100)

def minfilter_img(imdir,outdir):
    im = Image.open(imdir)
    out_filename = outdir
    out_img = im.filter(ImageFilter.MinFilter).save(out_filename, 'JPEG', quality=100)

def maxfilter_img(imdir,outdir):
    im = Image.open(imdir)
    out_filename = outdir
    out_img = im.filter(ImageFilter.MaxFilter).save(out_filename, 'JPEG', quality=100)
          
def blur_img(imdir,outdir):
    im = Image.open(imdir)
    out_filename = outdir
    out_img = im.filter(ImageFilter.BLUR).save(out_filename, 'JPEG', quality=100)

def edgeenhance_img(imdir,outdir):
    im = Image.open(imdir)
    out_filename = outdir
    im.filter(ImageFilter.EDGE_ENHANCE).save(out_filename, 'JPEG', quality=100)

def sharpen_img(imdir,outdir):
    im = Image.open(imdir)
    out_filename = outdir
    im.filter(ImageFilter.SHARPEN).save(out_filename, 'JPEG', quality=100)

def gray_img(imdir,outdir):    
    im = Image.open(imdir).convert('L')
    out_filename = outdir
    im.save(out_filename, 'JPEG', quality = 100)

def rotate_img(imdir,outdir,deg):    
    im = Image.open(imdir)
    out_filename = outdir
    im.rotate(deg).save(out_filename, 'JPEG', quality = 100)

def fliplr_img(imdir,outdir,deg):    
    im = Image.open(imdir)
    out_filename = outdir
    im.transpose(Image.FLIP_LEFT_RIGHT).save(out_filename, 'JPEG', quality = 100)

def fliplr_img(imdir,outdir):    
    im = Image.open(imdir)
    out_filename = outdir
    im.transpose(Image.FLIP_LEFT_RIGHT).save(out_filename, 'JPEG', quality = 100)

def equalize_img(imdir,outdir):    
    im = Image.open(imdir)
    out_filename = outdir
    ImageOps.equalize(im).save(out_filename, 'JPEG', quality = 100)

def autocontrast_img(imdir,outdir):    
    im = Image.open(imdir)
    out_filename = outdir
    ImageOps.autocontrast(im).save(out_filename, 'JPEG', quality = 100)

def crop_img(imdir,outdir,pct):    
    im = Image.open(imdir)
    width, height = im.size
    out_filename = outdir
    ImageOps.crop(im,int(width*pct)).save(out_filename, 'JPEG', quality = 100)

def expand_img(imdir,outdir,pct,color):    
    im = Image.open(imdir)
    width, height = im.size
    out_filename = outdir
    ImageOps.expand(im,int(width*pct), int(color)).save(out_filename, 'JPEG', quality = 100)

def invert_img(imdir,outdir):    
    im = Image.open(imdir)
    out_filename = outdir
    ImageOps.invert(im).save(out_filename, 'JPEG', quality = 100)

def posterize_img(imdir,outdir,bit):    
    im = Image.open(imdir)
    out_filename = outdir
    ImageOps.posterize(im,bit).save(out_filename, 'JPEG', quality = 100)
    
def saveas_jpg(imagedir):
    if imagedir[-3:] != "jpg":
        newname = os.path.splitext(imagedir)[0] + ".jpg"
        image = Image.open(imagedir).convert("RGB").save(newname, "JPEG", quality = 90)


		
		

# Image Augmentation script -- comment out portions you want to omit		
# NOTE: outdir should be an existing directory with 5 subfolders enumerated in "dirs"

dirs = ['heart', 'oblong', 'oval', 'round', 'square']

if (1):
    time_start = time.monotonic()

#NOTE: imagedir should contain sub-folders containing the pictures to be processed; outdir should be manually created with the same sub-folders as imagedir
	
    imagedir = "C:/Users/Adonis Tio/Jupyter/Google Images/celebs3_squared/"
    outdir = "C:/Users/Adonis Tio/Jupyter/Google Images/celebs3_augmented/"

    sub_dir = [q for q in pathlib.Path(imagedir).iterdir() if q.is_dir()]
    #print(len(sub_dir))
    for j in range(0,len(sub_dir)):
        images_dir = [p for p in pathlib.Path(sub_dir[j]).iterdir() if p.is_file()]
        #print(dir(images_dir)) #print(images_dir.__sizeof__()) #print(type(images_dir)) #print(len(images_dir))
        #print(len(images_dir))
        for i in range(0,len(images_dir)):
			#print(images_dir[i], type(images_dir[i])) #print(str(images[i])) #Image.open(images_dir[i]).show()
            #print("Processing ", images_dir[i], "...", end=' ', sep='') 
            
			outfile = outdir + dirs[j] + "/" + "square_" + os.path.split(str(images_dir[i]))[-1]
            make_img_square(images_dir[i], outfile)            
			
            #outfile = outdir + dirs[j] + "/" + "blur_" + os.path.split(str(images_dir[i]))[-1]
            #blur_img(images_dir[i], outfile)
            #outfile = outdir + dirs[j] + "/" + "minfilter_" + os.path.split(str(images_dir[i]))[-1]
            #minfilter_img(images_dir[i], outfile)
            #outfile = outdir + dirs[j] + "/" + "maxfilter_" + os.path.split(str(images_dir[i]))[-1]
            #maxfilter_img(images_dir[i], outfile)

            outfile = outdir + dirs[j] + "/" + "edgeenhance_" + os.path.split(str(images_dir[i]))[-1]
            edgeenhance_img(images_dir[i], outfile)
            #outfile = outdir + dirs[j] + "/" + "sharpen_" + os.path.split(str(images_dir[i]))[-1]
            #sharpen_img(images_dir[i], outfile)

            outfile = outdir + dirs[j] + "/" + "gray_" + os.path.split(str(images_dir[i]))[-1]
            gray_img(images_dir[i], outfile)

            outfile = outdir + dirs[j] + "/" + "rot-_" + os.path.split(str(images_dir[i]))[-1]
            rotate_img(images_dir[i], outfile, 10)
            outfile = outdir + dirs[j] + "/" + "rot+_" + os.path.split(str(images_dir[i]))[-1]
            rotate_img(images_dir[i], outfile, -10)

            #outfile = outdir + dirs[j] + "/" + "fliplr_" + os.path.split(str(images_dir[i]))[-1]
            #fliplr_img(images_dir[i], outfile)

            outfile = outdir + dirs[j] + "/" + "equalize_" + os.path.split(str(images_dir[i]))[-1]
            equalize_img(images_dir[i], outfile)
            outfile = outdir + dirs[j] + "/" + "autocontrast_" + os.path.split(str(images_dir[i]))[-1]
            autocontrast_img(images_dir[i], outfile)

            outfile = outdir + dirs[j] + "/" + "crop10pc_" + os.path.split(str(images_dir[i]))[-1]
            crop_img(images_dir[i], outfile, 0.1)
            outfile = outdir + dirs[j] + "/" + "expand10pc_" + os.path.split(str(images_dir[i]))[-1]
            expand_img(images_dir[i], outfile, 0.1, int(1))

            outfile = outdir + dirs[j] + "/" + "invert_" + os.path.split(str(images_dir[i]))[-1]
            invert_img(images_dir[i], outfile)
            outfile = outdir + dirs[j] + "/" + "posterize2_" + os.path.split(str(images_dir[i]))[-1]
            posterize_img(images_dir[i], outfile, 2)
            
    print("Runtime: (", "{0:.2f}".format(time.monotonic()-time_start), "s)", sep='')
