import numpy as np
import numpy
import theano

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, shared_y

def import_data(file_num = 2):
    gdata_file_name = "./game_data/gdata%04lu.bin" % file_num
    label_file_name = "./game_data/label%04lu.bin" % file_num
    print("Importing \"%s\"" % gdata_file_name)
    print("Importing \"%s\"" % label_file_name)
    game_data = np.fromfile(gdata_file_name, dtype=np.float32)
    label_data = np.fromfile(label_file_name, dtype=np.float32)
    num_boards = label_data.size
    game_data = game_data.reshape(num_boards, 3, 8, 8)

    data = (game_data, label_data)
    
    return shared_dataset(data)

if __name__ == "__main__":
    import_data()

