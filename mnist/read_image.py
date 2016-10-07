import os
#from PIL import Image
import numpy as np
import struct

_author_ = 'Zhu Peihao'


# Function of reading and saving information from binary files of training set and test set.
# Save the pixel gray values as arrays in .npy files.
def read_image(filename, save_filename):
    """

    :param filename: the binary file of images to be read
    :param save_filename: the .npy file to save the arrays of pixels from the binary file
    """
    # Open the file "train-images.idx3-ubyte" or "t10k-images.idx3-ubyte".
    f = open(filename, 'rb')
    index = 0
    buf = f.read()
    f.close()

    # Access to basic information. Images means the number of images.
    magic, images, rows, columns = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')

    # List for storing pixel information
    im_list = []

    for i in xrange(images):

        '''
        # Use the PIL to save the pixels as pictures
        image = Image.new('L', (columns, rows))

        for x in xrange(rows):
            for y in xrange(columns):
                image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
                index += struct.calcsize('>B')

        #print 'save' + str(i) + 'image'
        #image.save('train/' + str(i) + '.png')
        '''

        # The image is 28 * 28 pixels, a total of 784 B
        im = struct.unpack_from('>784B', buf, index)
        index += struct.calcsize('>784B')
        im = np.array(im)
        im_list.append(im)

    # Convert a list into an array
    im_list = np.array(im_list)
    np.save(save_filename, im_list)
    print 'save' + save_filename + 'successfully'


# Save the labels as an array in .npy files.
def read_label(filename, save_filename):
    """

    :param filename: the binary file of labels to be read
    :param save_filename: the .npy file to save the array of labels
    """
    # Open the file "train-labels.idx1-ubyte" or "test_label.npy".
    f = open(filename, 'rb')
    index = 0
    buf = f.read()
    f.close()

    magic, labels = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')

    # List for storing label information
    label_array = [0] * labels
    for i in xrange(labels):
        # The label is a number from 0 to 9, accounted for one B
        label_array[i] = int(struct.unpack_from('>B', buf, index)[0])
        index += struct.calcsize('>B')

    # Convert a list into an array
    label_array = np.array(label_array)
    np.save(save_filename, label_array)
    print 'save' + save_filename + 'successfully'


# Test call
if __name__ == '__main__':
    read_image('train-images.idx3-ubyte', 'train_images.npy')
    read_image('t10k-images.idx3-ubyte', 'test_images.npy')
    read_label('train-labels.idx1-ubyte', 'train_label.npy')
    read_label('t10k-labels.idx1-ubyte', 'test_label.npy')


