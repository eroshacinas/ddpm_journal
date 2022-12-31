# example of calculating the frechet inception distance in Keras
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize

class Gan_Evaluator():
  def __init__(self):
    self.model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

  
  def load_dataset(self, source_dir, img_size=128, isRGB=True):
    import glob
    import os
    from PIL import Image
    import numpy as np
    import cv2

    dir_list = glob.glob(source_dir + '/*.jpg')
    img_arr = np.zeros(len(dir_list))
    img_list = list()  

    for file in dir_list:
      if isRGB:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert from BGR to RGB

      else:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE) #open in grayscale
      
      
      #crop image if it's out of shape
      if(img.shape[0] != img_size or img.shape[1] != img_size): 
        img = crop_single_image(path=file, target_size=img_size)

      img_list.append(img)

    
    return np.array(img_list)


  # scale an array of images to a new size
  def scale_images(self, images, new_shape):
    images_list = list()
    for image in images:
      # resize with nearest neighbor interpolation
      new_image = resize(image, new_shape, 0)
      # store
      images_list.append(new_image)
    return asarray(images_list)

  # calculate frechet inception distance
  def calculate_fid(self, images1, images2):
    # calculate activations
    act1 = self.model.predict(images1)
    act2 = self.model.predict(images2)

    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
      covmean = covmean.real
    
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


  def evaluate_gan(self, image1_source, image2_source):
    self.images1 = self.load_dataset(image1_source)
    self.images2 = self.load_dataset(image2_source)

    print('Prepared', self.images1.shape, self.images2.shape)
    
    # convert integer to floating point values
    self.images1 = self.images1.astype('float32')
    self.images2 = self.images2.astype('float32')

    # resize images
    self.images1 = self.scale_images(self.images1, (299,299,3))
    self.images2 = self.scale_images(self.images2, (299,299,3))


    print('Scaled', self.images1.shape, self.images2.shape)
    # pre-process images
    self.images1 = preprocess_input(self.images1)
    self.images2 = preprocess_input(self.images2)

    # fid between images1 and images2
    fid = self.calculate_fid(self.images1, self.images2)
    print('FID (different): %.3f' % fid)

    return fid



def crop_single_image(path, target_size):
  from PIL import Image
  import os
  import numpy as np

  imgPrime = Image.open(path)
  imgPrime = imgPrime.crop((0,0, target_size, target_size)) #crop image 
  img = np.array(imgPrime) #convert back to numpy image

  return img
