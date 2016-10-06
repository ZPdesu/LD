import os
import numpy as np
import struct
#from PIL import Image


def read_image(filename, save_filename):
    f = open(filename, 'rb')
    index = 0
    buf = f.read()
    f.close()

    magic, images, rows, columns = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')

    im_list = []

    for i in xrange(images):

        '''
        image = Image.new('L', (columns, rows))

        for x in xrange(rows):
            for y in xrange(columns):
                image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
                index += struct.calcsize('>B')

        #print 'save' + str(i) + 'image'
        #image.save('train/' + str(i) + '.png')
        '''

        im = struct.unpack_from('>784B', buf, index)
        index += struct.calcsize('>784B')
        im = np.array(im)
        im_list.append(im)

    im_list = np.array(im_list)
    np.save(save_filename, im_list)
    print 'save' + save_filename + 'successfully'


def read_label(filename, save_filename):
    f = open(filename, 'rb')
    index = 0
    buf = f.read()
    f.close()

    magic, labels = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')

    label_array = [0] * labels
    for i in xrange(labels):
        label_array[i] = int(struct.unpack_from('>B', buf, index)[0])
        index += struct.calcsize('>B')

    label_array = np.array(label_array)
    np.save(save_filename, label_array)
    print 'save' + save_filename + 'successfully'


if __name__ == '__main__':
    read_image('train-images.idx3-ubyte', 'train_images.npy')
    read_image('t10k-images.idx3-ubyte', 'test_images.npy')
    read_label('train-labels.idx1-ubyte', 'train_label.npy')
    read_label('t10k-labels.idx1-ubyte', 'test_label.npy')


