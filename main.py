from glob import glob
from os.path import join
import numpy as np
import cv2
import matplotlib.pyplot as plt
from register_image import register_image

if __name__ == '__main__':

  pagenumber = 1
  raw_images_dir = 'img_raw/*'
  aligned_images_dir = 'img_align'
  orig_images_dir = 'img_orig'
  
  # read reference image
  refFilename = glob(join(raw_images_dir, 'Suitability Form - Nationwide - BLANK-'+str(pagenumber)+'.png'))[0]
  print("Reading reference image : ", refFilename)
  imRef = cv2.imread(refFilename, cv2.IMREAD_COLOR)

  # get path of all the image files that we are gonna register
  allFiles = glob(join(raw_images_dir, '*-'+str(pagenumber)+'.png'))
  for imFilename in allFiles:

    # Read image to be aligned
    print("Reading image to align : ", imFilename)
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    # align the image
    imReg, h = register_image(im, imRef)
    imReg = imReg[:im.shape[0], :im.shape[1], :]

    # add space for annotation on top of image ("original" or "aligned")
    blank = 255*np.ones_like(imReg)[:160,:,:]
    imReg = np.append(blank, imReg, axis=0)
    im = np.append(blank, im, axis=0)

    # add the actual annotation
    textorg = (int(np.floor(imReg.shape[1]/3)), 120)
    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(im, 'Original', textorg, font, fontScale=3, color=[150,150,150], thickness=10)
    cv2.putText(imReg, 'Aligned', textorg, font, fontScale=3, color=[150,150,150], thickness=10)

    # concatenate both iaages together
    imPair = np.append(im, imReg, axis=1)

    # write image pair to disk
    outFilename = join(aligned_images_dir, imFilename.split('/')[-1][:-4]+"-aligned.jpg")
    print("Saving aligned image : ", outFilename)
    cv2.imwrite(outFilename, imPair)

