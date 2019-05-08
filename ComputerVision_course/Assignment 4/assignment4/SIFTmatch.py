from PIL import Image, ImageDraw
import numpy as np
import csv
import math

def ReadKeys(image):
    """Input an image and its associated SIFT keypoints.

    The argument image is the image file name (without an extension).
    The image is read from the PGM format file image.pgm and the
    keypoints are read from the file image.key.

    ReadKeys returns the following 3 arguments:

    image: the image (in PIL 'RGB' format)

    keypoints: K-by-4 array, in which each row has the 4 values specifying
    a keypoint (row, column, scale, orientation).  The orientation
    is in the range [-PI, PI] radians.

    descriptors: a K-by-128 array, where each row gives a descriptor
    for one of the K keypoints.  The descriptor is a 1D array of 128
    values with unit length.
    """
    im = Image.open(image+'.pgm').convert('RGB')
    keypoints = []
    descriptors = []
    first = True
    with open(image+'.key','r') as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC,skipinitialspace = True)
        descriptor = []
        for row in reader:
            if len(row) == 2:
                assert first, "Invalid keypoint file header."
                assert row[1] == 128, "Invalid keypoint descriptor length in header (should be 128)."
                count = row[0]
                first = False
            if len(row) == 4:
                keypoints.append(np.array(row))
            if len(row) == 20:
                descriptor += row
            if len(row) == 8:
                descriptor += row
                assert len(descriptor) == 128, "Keypoint descriptor length invalid (should be 128)."
                #normalize the key to unit length
                descriptor = np.array(descriptor)
                descriptor = descriptor / math.sqrt(np.sum(np.power(descriptor,2)))
                descriptors.append(descriptor)
                descriptor = []
    assert len(keypoints) == count, "Incorrect total number of keypoints read."
    print ("Number of keypoints read:", int(count))
    return [im,keypoints,descriptors]

def AppendImages(im1, im2):
    """Create a new image that appends two images side-by-side.

    The arguments, im1 and im2, are PIL images of type RGB
    """
    im1cols, im1rows = im1.size
    im2cols, im2rows = im2.size
    im3 = Image.new('RGB', (im1cols+im2cols, max(im1rows,im2rows)))
    im3.paste(im1,(0,0))
    im3.paste(im2,(im1cols,0))
    return im3

def DisplayMatches(im1, im2, matched_pairs):
    """Display matches on a new image with the two input images placed side by side.

    Arguments:
     im1           1st image (in PIL 'RGB' format)
     im2           2nd image (in PIL 'RGB' format)
     matched_pairs list of matching keypoints, im1 to im2

    Displays and returns a newly created image (in PIL 'RGB' format)
    """
    im3 = AppendImages(im1,im2)
    offset = im1.size[0]
    draw = ImageDraw.Draw(im3)
    for match in matched_pairs:
        draw.line((match[0][1], match[0][0], offset+match[1][1], match[1][0]),fill="red",width=2)
    im3.show()
    return im3

def match(image1,image2):
    """Input two images and their associated SIFT keypoints.
    Display lines connecting the first 5 keypoints from each image.
    Note: These 5 are not correct matches, just randomly chosen points.

    The arguments image1 and image2 are file names without file extensions.

    Returns the number of matches displayed.

    Example: match('scene','book')
    """
    im1, keypoints1, descriptors1 = ReadKeys(image1)
    im2, keypoints2, descriptors2 = ReadKeys(image2)
    #
    # REPLACE THIS CODE WITH YOUR SOLUTION (ASSIGNMENT 5, QUESTION 3)
    #
    #Generate five random matches (for testing purposes)
    # initialize values
    matched_pairs = []
    threshold = 0.5
    # calculating dot product
    # loop through each elemtnt in descriptors1
    for i in range(len(descriptors1)):

        # empty dot product array to be filled
        arr_dots = []

        # loop through each element in descriptors2
        for j in range (len(descriptors2)):

            # compute the dot product of a element in descriptor1 to correspond element in descriptor2
            dot = np.dot(descriptors1[i], descriptors2[j])
            #append each of the dot products to the array of dot products
            arr_dots.append(dot)

        # empty arrays for angles to be filled
        arr_angles = []

        # loop through the array of dot products
        for dotpr in range (len(arr_dots)):

            # compute the angle between two corresponding points in descriptors
            arr_angles.append(math.acos(arr_dots[dotpr]))

        # to find the two first best matches, we need to sort the array of angles from the smallest one to largest one
        sort_arr_angles = sorted(arr_angles)
        best_angle = sort_arr_angles[0]
        second_best_angle = sort_arr_angles[1]

        # compute the ratio to eliminate the false matches
        ratio = best_angle/second_best_angle
        index = 0

        # A match should be selected only if this ratio is below a threshold
        if (threshold >= ratio):

            # calculate the index of the matched points in the array of angles to find the point in the second image
            index  = arr_angles.index(best_angle)
            #match the point in the first descriptor with corresponding best match in the second descriptor
            matched_pairs.append([keypoints1[i], keypoints2[index]])


    # initalize values that is used in RANSAC
    diff_orientation_btw_keypoints = 20 # plus or minus
    diff_scale = 0.8
    ransac_all = []

    # Repeat the random selection 10 times and then select the largest consistent subset that was found.
    for x in range(10):
        # For each RANSAC iteration you will select just one match at random,
        # and then check all the other matches for consistency with it.
        ransacc = []
        rnd_match = np.random.randint( len(matched_pairs))
        random_matchpairs = matched_pairs[rnd_match]

        # calculate change in scale and orientation btw pairs
        diff_orientation = random_matchpairs[0][3] - random_matchpairs[1][3]
        diff_scale = abs(random_matchpairs[0][2] - random_matchpairs[1][2])


        # then check all the other matches for consistency with it
        for m_pairs in range( len(matched_pairs)):
            m_pair = matched_pairs[m_pairs]
            diff_orientation_allpairs = m_pair[0][3] - m_pair[1][3]
            diff_in_angle = abs(math.degrees(diff_orientation) + 360 - math.degrees(diff_orientation_allpairs) + 360) % 180

            # if the change in angle fits our specfication and check for consistancy
            if (diff_in_angle <= diff_orientation_btw_keypoints):

                # then we compare scale change and consistancy
                diff_scale_allpairs = abs(m_pair[0][2] - m_pair[1][2])
                diff_scale_lower = min(diff_scale, diff_scale_allpairs)
                diff_scale_upper = max(diff_scale, diff_scale_allpairs)
                change_of_scale = diff_scale_upper * diff_scale

                # if it is consistant enough, we add it to the current array
                if (change_of_scale <= diff_scale_lower):
                    ransacc.append(m_pair)

                # consistancy check failed (scale), skip this item
                else:
                    continue;
            # consistancy check failed (angle), skip item
            else:
                continue;
        # select the largest consistent subset that was found.
        if (len(ransacc) > len(ransac_all)):

           ransac_all = ransacc;

    #
    # END OF SECTION OF CODE TO REPLACE
    #
    im3 = DisplayMatches(im1, im2, ransac_all)
    #im3 = DisplayMatches(im1, im2, matched_pairs)

    return im3


#Test run
match('library','library2')
#match('scene','basmati')
#match('scene', 'book')
