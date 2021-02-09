# Sort-By-Face
This is an application with which you can either sort all the pictures by faces from a corpus of photos or retrieve all your photos from the corpus  
by submitting a picture of yours.

# How it works
- Given a corpus of photos inside a directory this application first detects the faces in the photos and runs a Convolutional Neural Network to  
generate 128-Dimensional embeddings. 
- These embeddings are then used in a graph based clustering algorithm called 'Chinese Whispers'.  
- The clustering algorithm assigns a cluster to each individual identified by it.  
- After the algorithm the images are copied into seperate directories corresponding to their clusters.
- For a person who wants to retrieve only his images, only the images which are in the same cluster as the picture submitted by the user is copied.

# References
This project is inspired by the ideas presented in the following papers

[[1] ](https://repository.tudelft.nl/islandora/object/uuid:a9f82787-ac3d-4ff1-8239-4f3c1c6414b9)Roy Klip. Fuzzy Face Clustering For Forensic Investigations

[[2] ](https://www.hindawi.com/journals/cin/2019/6065056/)Chang L, Pérez-Suárez A, González-Mendoza M. Effective and Generalizable Graph-Based Clustering for Faces in the Wild.

[[3] ](https://www.researchgate.net/publication/228670574_Chinese_whispers_An_efficient_graph_clustering_algorithm_and_its_application_to_natural_language_processing_problems) Biemann, Chris. (2006). Chinese whispers: An efficient graph clustering algorithm and its application to natural language processing problems. 