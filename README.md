# InformationRetrieval
Information Retrieval, Extraction and Integration course assignments
## Toy CBIR
In this assignment, a content-based clothing image retrieval system is proposed.

This system is made up of a main collection of 110 images (images/index) and 5 query images (images/search). 
It has been used: OpenCV library to compute the image features (descriptors) and match them (Brute-force descriptor matcher), Pandas library to return the resulting rankings in dataframe form, and the IPython library for displaying dataframes in HTML form. 

The process followed to generate the image rankings is as follows:

1.	Read images in grey scale.
2.	All image descriptors are obtained (database images and query images).
3.	Get the matching descriptors between the database images descriptors and query images descriptors (only descriptors of two images at a time) and get their distances. 
4.	The total distance between the images is computed (from matching descriptors distances).
5.	The similarity between the images is computed (from the total distance).
6.	Three rankings are returned (one per approach) with all similarities from highest to lowest.

To run the CBIR system, simply run these commands:
```
pip3 install --upgrade pip
pip3 install -r requirements.txt
python3 clothing_cbir.py
```

The three returned rankings are found inside the rankings folder. Screenshots of the results by approach and type of clothing have also been included in the folder results_by_algorithm_and_clothing.
