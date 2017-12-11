# Starter code for uesr-based collaborative filtering
# Complete the function user_based_cf below. Do not change it arguments and return variables. 
# Do not change main() function, 

# import modules you need here.
import sys
import csv
import numpy as np
import utils


def user_matrix(userids, movieids, ratings, numOfUsers, numOfItems):
    """
    Construct a matrix *M* of *numOfUsers*-by-*numOfItems*, where each row is a user vector.
    For example, row 0 is the user vector of the user whose id = 1. (0, 0) is the rating of user 1 to movie 1.
    """
    
    M = np.zeros((numOfUsers, numOfItems))
    flat_pos = np.ravel_multi_index((userids - 1, movieids - 1), M.shape)
    np.put(M, flat_pos, ratings)
    return M

def mode(arr):
    """ Find the element with most occurrence, or return the first element in arr. """
    values, counts = np.unique(arr, return_counts=True)
    same_freq = False
    if len(np.where(counts > 1)[0]) == 0:
        return arr[0]
    else:
        ind = np.argmax(counts)
        return values[ind]

def user_based_cf(datafile, userid, movieid, distance, k, iFlag, numOfUsers, numOfItems):
    '''
    build user-based collaborative filter that predicts the rating 
    of a user for a movie.
    This function returns the predicted rating and its actual rating.
    
    Parameters
    ----------
    <datafile> - a fully specified path to a file formatted like the MovieLens100K data file u.data 
    <userid> - a userId in the MovieLens100K data
    <movieid> - a movieID in the MovieLens 100K data set
    <distance> - a Boolean. If set to 0, use Pearson's correlation as the distance measure. If 1, use Manhattan distance.
    <k> - The number of nearest neighbors to consider
    <iFlag> - A Boolean value. If set to 0 for user-based collaborative filtering, 
    only users that have actual (ie non-0) ratings for the movie are considered in your top K. 
    For user-based, use only movies that have actual ratings by the user in your top K. 
    If set to 1, simply use the top K regardless of whether the top K contain actual or filled-in ratings.
    <numOfUsers> - the number of users in the dataset 
    <numOfItems> - the number of items in the dataset
    (NOTE: use these variables (<numOfUsers>, <numOfItems>) to build user-rating matrix. 
    DO NOT USE any CONSTANT NUMBERS when building user-rating matrix. We already set these variables in the main function for you.
    The size of user-rating matrix in the test case for grading could be different from the given dataset. )

    returns
    -------
    trueRating: <userid>'s actual rating for <movieid>
    predictedRating: <userid>'s rating predicted by collaborative filter for <movieid>


    AUTHOR: Shiyu Luo (This is where you put your name)
    '''
    # read file
    u_data = csv.reader(open(datafile, 'rb'), delimiter='\t')
    columns = list(zip(*u_data))
    # column 1: user id
    col1 = np.array(columns[0]).astype(np.int)
    # column 2: item id
    col2 = np.array(columns[1]).astype(np.int)    
    # column 3: ratings
    col3 = np.array(columns[2]).astype(np.int)
    
    # construct a 943*1682 matrix, where u_mat(i, j) is the rating of movie j+1 given by user i+1
    u_mat = user_matrix(col1, col2, col3, numOfUsers, numOfItems)
    # fetch true rating from the matrix
    trueRating = u_mat[userid - 1, movieid - 1]
    
    # finds the k nearest user vectors 
    neighbors = utils.knn(mat=u_mat, target_row=userid - 1, nonzero_col=movieid - 1, metric=distance, k=k, iFlag=iFlag)
    
    # ratings given by these neighbors
    ratings = neighbors[:, movieid - 1]
    # what is the most common rating?
    # if there is no such rating, simply returns the nearest.
    # no good reason
    predictedRating = mode(ratings)

    return trueRating, predictedRating
    

def main():
    datafile = sys.argv[1]
    userid = int(sys.argv[2])
    movieid = int(sys.argv[3])
    distance = int(sys.argv[4])
    k = int(sys.argv[5])
    i = int(sys.argv[6])
    numOfUsers = 943
    numOfItems = 1682
    
    trueRating, predictedRating = user_based_cf(datafile, userid, movieid, distance, k, i, numOfUsers, numOfItems)
    print 'userID:{} movieID:{} trueRating:{} predictedRating:{} distance:{} K:{} I:{}'.format(userid, movieid, trueRating, predictedRating, distance, k, i)
    

if __name__ == "__main__":
    main()
    