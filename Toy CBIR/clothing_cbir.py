import cv2 as cv
import os
import pandas as pd
from IPython.display import HTML

index_dir_path = os.path.abspath('images/index/') 
search_dir_path = os.path.abspath('images/search/')
search_dir = os.listdir(search_dir_path)
index_dir = os.listdir(index_dir_path)

def get_sift_image_descriptors(search_img, idx_img):
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # Find keypoints and compute descriptors with SIFT
    _, search_des_sift = sift.detectAndCompute(search_img,None)
    _, idx_des_sift = sift.detectAndCompute(idx_img,None)
    return search_des_sift, idx_des_sift 

def get_orb_image_descriptors(search_img, idx_img):
    # Initiate ORB detector
    orb = cv.ORB_create()
    # Find keypoints and compute descriptors with ORB
    _, search_des_orb = orb.detectAndCompute(search_img,None)
    _, idx_des_orb = orb.detectAndCompute(idx_img,None)
    return search_des_orb,idx_des_orb

def get_orb_sift_image_descriptors(search_img, idx_img):
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # Initiate ORB detector
    orb = cv.ORB_create()
    # Find keypoints with ORB
    search_kp_orb = orb.detect(search_img, None)
    idx_kp_orb = orb.detect(idx_img, None)
    # Compute descriptors with SIFT
    _, search_des_sift = sift.compute(search_img, search_kp_orb)
    _, idx_des_sift = sift.compute(idx_img, idx_kp_orb)
    return search_des_sift, idx_des_sift

def get_similarity_from_desc(approach, search_desc, idx_desc):
    if approach == 'sift' or approach == 'orb_sift':
        # BFMatcher with euclidean distance
        bf = cv.BFMatcher()
    else:
        # BFMatcher with hamming distance
        bf = cv.BFMatcher(cv.NORM_HAMMING)
    matches = bf.match(search_desc, idx_desc)
    # Distances between search and index features that match
    distances = [m.distance for m in matches]
    # Distance between search and index images
    distance = sum(distances) / len(distances)
    # If distance == 0 -> similarity = 1
    similarity = 1 / (1 + distance)
    return similarity
    
def get_ranking_from_desc(approach):
    dfs = []
    for search_img in search_dir:
        df = pd.DataFrame(columns=['search_image', 'index_image', 'similarity_score'])
        similarities = []
        for idx_img in index_dir:
           # Read images in gray scale
           search = cv.imread(os.path.join(search_dir_path, search_img) , cv.IMREAD_GRAYSCALE)
           idx = cv.imread(os.path.join(index_dir_path, idx_img), cv.IMREAD_GRAYSCALE)
           if approach == 'sift':
               img_descriptors = get_sift_image_descriptors(search,idx)
           elif approach == 'orb':
               img_descriptors = get_orb_image_descriptors(search,idx)
           else:
               img_descriptors = get_orb_sift_image_descriptors(search,idx)
           # Get similarity scores given a search image
           similarities.append(get_similarity_from_desc(approach, img_descriptors[0], img_descriptors[1]))
        df['search_image'] = [search_img] * len(index_dir)
        df['index_image'] = index_dir
        df['similarity_score'] = similarities
        df = df.sort_values(by='similarity_score', ascending=False)
        # Select top 20 matches for every search image
        df = df.head(20)
        dfs.append(df)
    # Build ranking with similarity scores for all search images
    ranking = pd.concat(dfs).reset_index(drop=True)
    return ranking

def path_to_image_html(path):
    return '<img src="'+ path + '" style=max-height:90px;"/>'

def write_html_ranking(approach, ranking):
    # Display ranking in HTML format
    html_ranking = ranking.copy(deep = True)
    html_ranking['search_image'] = ranking['search_image'].apply(lambda x: path_to_image_html(os.path.join(search_dir_path, x)))
    html_ranking['index_image'] = ranking['index_image'].apply(lambda x: path_to_image_html(os.path.join(index_dir_path, x)))
    html = HTML(html_ranking.to_html(escape=False))
    with open('rankings/' + approach + "_ranking.html", "w") as file:
        file.write(html.data)

if __name__ == "__main__":
    
    #######################################################################
    # First approach: SIFT + Euclidean distance
    #######################################################################
    sift_ranking = get_ranking_from_desc('sift')
    write_html_ranking('sift', sift_ranking)

    #######################################################################
    # Second approach: ORB + Hamming distance
    #######################################################################
    orb_ranking = get_ranking_from_desc('orb')
    write_html_ranking('orb', orb_ranking)

    ########################################################################
    # Third approach: ORB keypoints + SIFT descriptors + Euclidean distance
    ########################################################################
    combined_ranking = get_ranking_from_desc('orb_sift')
    write_html_ranking('combined', combined_ranking)
