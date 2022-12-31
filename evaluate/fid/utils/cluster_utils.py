# Import library
from clustimage import Clustimage
import glob
import os
import matplotlib.pyplot as plt

#load file paths in folder and return a list

#filepath = list of paths to images
def cluster_images(filepath):
  # init with PCA method and all default parameters
  cl = Clustimage(method='pca')


  # Import data. This can either by a list with pathnames or NxM array.
  X = cl.import_data(filepath)

  # Extract features with the initialized method (PCA in this case)
  Xfeat = cl.extract_feat(X)
  # Embedding using tSNE
  xycoord = cl.embedding(Xfeat)

  # Cluster 
  labels = cl.cluster(cluster='agglomerative',
                      evaluate='silhouette',
                      metric='euclidean',
                      linkage='ward',
                      min_clust=2,
                      max_clust=25,
                      cluster_space='high')

  return cl


def plot_cluster_graphs(cl, #actual class of cluster model
                plot_silh=True, #plot Silhouette
                plot_pca=True, #plot pca 
                plot_dendo=False, #plot dendogram
                plot_unique=True, #plot unique images per scatter
                plot_scatter=False, #plot scatterplot
                plot_images_per_cluster=False #plot all imaegs per scatter
                ):
  # Silhouette plots
  if plot_silh:
    cl.clusteval.plot()
    cl.clusteval.scatter(cl.results['xycoord'])

  # PCA explained variance plot
  if plot_pca:
    cl.pca.plot()

  # Dendrogram
  if plot_dendo:
    cl.dendrogram()

  # Plot unique image per cluster
  if plot_unique:
    cl.plot_unique(img_mean=False)

  # Scatterplot
  if plot_scatter:
    cl.scatter(zoom=0.5, img_mean=False)
    cl.scatter(zoom=None, img_mean=False)

  # Plot images per cluster or all clusters
  if plot_images_per_cluster:
    cl.plot(cmap='binary') #plot all images in all clusters


def extract_filepaths_from_cluster(cl, cluster_number):
  # Extracting images that belong to cluster label=0:
  Iloc = cl.results['labels']== cluster_number
  cluster_pathnames = cl.results['pathnames'][Iloc] #get index of images belonging to cluster 0

  # Extracting xy-coordinates for the scatterplot for cluster 0:
  import matplotlib.pyplot as plt
  xycoord = cl.results['xycoord'][Iloc]
  plt.scatter(xycoord[:,0], xycoord[:,1])

  return cluster_pathnames


def file_paths_to_list(source, file_type='*'): #return everything in a folder by default  
  if file_type == '*':
    image_path_list = glob.glob(os.path.join(source, file_type))

  else:
    image_path_list = glob.glob(os.path.join(source, '*.'+file_type))

  return image_path_list



def separate_paths(pathlist, real_dir, generated_dir):
  real_image_pathlist = list()
  generated_image_pathlist = list()

  for path in pathlist:
    if real_dir in path:
      real_image_pathlist.append(path)

    elif generated_dir in path:
      generated_image_pathlist.append(path)

  
  return real_image_pathlist, generated_image_pathlist
