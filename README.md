# Sort-By-Face
This is an application with which you can either sort all the pictures by faces from a corpus of photos or retrieve all your photos from the corpus  
by submitting a picture of yours.

# Instructions
- Put the directory where the folders are located into the project folder.
- Run `python embedder.py -src /path/to/images`. This command utilizes all the cores in the system for parallel processing.
- In case you want to reduce the number of parallel processes, run `python embedder.py -src /path/to/images --processes number-of-processes`.
- The above command then calculates all the embeddings for the faces in the pictures. NOTE: It takes a significant amount of time for large directories.
- The embeddings are saved in a pickle file called `embeddings.pickle`.
## Sort an entire corpus of photos
- Run `python sort_images.py`. This runs the clustering algorithm with the default parameters of threshold and iterations for the clustering algorithm.
- If you want to tweak the parameters, run `python sort_images.py -t threshold -itr num-iterations` to alter the threshold and iterations respectively.
- Once the clustering is finished all the images are stored into a folder called `Sorted-pictures`. Each subdirectory in it corresponds to the unique person identified.

## Get pictures of a single person from the corpus.
- To get pictures of a single person you will need to provide a picture of that person. It is recommended that the picture clears the following requirements
for better results:
    - Image must have width and height greater than 160px.
    - Image must consist of only one face (The program is exited when multiple faces are detected)
    - Image must be preferably well lit and recognizable by a human.
- Run `python get_individual.py -src /path/to/person's/image -dest /path/to/copy/images`.
- This script also allows to tweak with the parameters with the same arguments as mentioned before.
- Once clustering is done all the pictures are copied into the destination

# How it works
- Given a corpus of photos inside a directory this application first detects the faces in the photos and runs a Convolutional Neural Network to  
generate 128-Dimensional embeddings. 
- These embeddings are then used in a graph based clustering algorithm called 'Chinese Whispers'.  
- The clustering algorithm assigns a cluster to each individual identified by it.  
- After the algorithm the images are copied into seperate directories corresponding to their clusters.
- For a person who wants to retrieve only his images, only the images which are in the same cluster as the picture submitted by the user is copied.

## Model used for embedding extraction
The project uses a model which was first introduced in this [paper](https://arxiv.org/abs/1503.03832). It uses a keras model converted from 
David Sandberg's implementation in [this](https://github.com/davidsandberg/facenet) repository.  
In particular it uses the model with the name `20170512-110547` which was converted using [this](https://github.com/nyoki-mtl/keras-facenet/blob/master/notebook/tf_to_keras.ipynb) script.

![](assets\triplet-loss.png)

All the facenet models are trained using a loss called triplet loss. This loss ensures that the model gives closer embeddings for same people and farther embeddings for different people.  
The models are trained on a huge amount of images out of which triplets are generated.

# Evaluation of clustering algorithm.
The notebook 
On testing on the Labeled Faces in the Wild dataset the following results were obtained. (threshold = 0.67, iterations=30)
- **Precision**: 0.89
- **Recall**: 0.99
- **F-measure**: 0.95
- **Clusters formed**: 6090 (5749 unique people in the dataset)

The LFW dataset has many images containing more than one face but only has a single label. This can have an effect on the evaluation metrics and the clusters formed. These factors have been discussed in detail in the notebook.

For example by running the script `get_individual.py` and providing a photo of George Bush will result in some images like this.   
In Layman terms we have gathered all the 'photobombs' of George Bush in the datset also because all the labels for the photo correspond to a different person.  
  
<img src="assets\photobomb.jpg" width=450px> 

# References
This project is inspired by the ideas presented in the following papers

[[1] ](https://repository.tudelft.nl/islandora/object/uuid:a9f82787-ac3d-4ff1-8239-4f3c1c6414b9)Roy Klip. Fuzzy Face Clustering For Forensic Investigations

[[2] ](https://www.hindawi.com/journals/cin/2019/6065056/)Chang L, Pérez-Suárez A, González-Mendoza M. Effective and Generalizable Graph-Based Clustering for Faces in the Wild.

[[3] ](https://www.researchgate.net/publication/228670574_Chinese_whispers_An_efficient_graph_clustering_algorithm_and_its_application_to_natural_language_processing_problems) Biemann, Chris. (2006). Chinese whispers: An efficient graph clustering algorithm and its application to natural language processing problems.  
[[4] ](https://arxiv.org/abs/1503.03832)Florian Schroff and Dmitry Kalenichenko and James Philbin (2015). FaceNet, a Unified Embedding for Face Recognition and Clustering