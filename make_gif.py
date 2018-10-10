import imageio
import numpy as np
import os
from glob import glob
from os.path import join
import sys
import random
import PIL.Image as Image

path = sys.argv[1]
if len(sys.argv)>2:
  outFile = sys.argv[2]
else:
  outFile = 'my_gif.gif'

print('==============> Making gif from files in '+path)

list_images = glob(join(path, '*'))
random.shuffle(list_images)
images = []
count = 1
for filename in list_images:
  img = Image.open(filename)
  width, height = img.size
  img = img.resize((int(np.floor(x/5)) for x in (width,height)), Image.ANTIALIAS)
  img = np.array(img)
  images.append(img)
  print('Image '+str(count)+' of '+str(len(list_images)))
  count+=1

print('Combining images into gif')
imageio.mimsave(outFile, images, 'GIF', duration=.3)
print('gif saved in '+str(outFile))
