from .classes import *
from .preprocess_data import *
import pdb


def generate_mean_pixel_intensities_for_controls():
    try:
        mean_pixel_dict_controls = pickle.load(open('intermediate/mean_pixel_dict_controls.py','rb'))
        return mean_pixel_dict_controls
    except:
        mean_pixel_dict_controls = {}

        for ID in negative_control_IDs:
            print(ID)
            a = Assay(ID, 'processed/CleanNegativeControls/')
            a.read_rawimages()
            mean_red = np.mean([np.mean(image[:,:,0].flatten(), dtype=np.float32) for image in a.images], dtype=np.float32)
            mean_green = np.mean([np.mean(image[:,:,1].flatten(), dtype=np.float32) for image in a.images], dtype=np.float32)
            mean_pixel_dict_controls[ID] = (mean_red,mean_green)

        for ID in positive_control_IDs:
            print(ID)
            a = Assay(ID, 'processed/CleanPositiveControls/')
            a.read_rawimages()
            mean_red = np.mean([np.mean(image[:,:,0].flatten(), dtype=np.float32) for image in a.images], dtype=np.float32)
            mean_green = np.mean([np.mean(image[:,:,1].flatten(), dtype=np.float32) for image in a.images], dtype=np.float32)
            mean_pixel_dict_controls[ID] = (mean_red,mean_green)
            pickle.dump(mean_pixel_dict_controls, open('intermediate/mean_pixel_dict_controls.py','wb'))
        return mean_pixel_dict_controls

def generate_mean_pixel_intensities_for_IXClean():
    try:
        mean_pixel_dict_IXClean = pickle.load(open('intermediate/mean_pixel_dict_IXClean.py','rb'))
        return mean_pixel_dict_IXClean
    except:
        mean_pixel_dict_IXClean = {}
        for ID in IXimage_IDs:
            print(ID)
            a = Assay(ID, 'processed/IXClean/')
            a.read_rawimages()
            mean_red = np.mean([np.mean(image[:,:,0].flatten(), dtype=np.float32) for image in a.images], dtype=np.float32)
            mean_green = np.mean([np.mean(image[:,:,1].flatten(), dtype=np.float32) for image in a.images], dtype=np.float32)
            mean_pixel_dict_IXClean[ID] = (mean_red,mean_green)

            pickle.dump(mean_pixel_dict_IXClean, open('intermediate/mean_pixel_dict_IXClean.py','wb'))
        return mean_pixel_dict_IXClean
    
def display_pixel_cluster(mean_pixel_dict_controls):
    pixels_intensity_labels = [(mean_pixel_dict_controls[ID][0], mean_pixel_dict_controls[ID][1], ID in total_valid_postiveIDs) for ID in mean_pixel_dict_controls]

    pixel_positive_labels = [x[0:2] for x in pixels_intensity_labels if x[-1] == True]
    pixel_negative_labels = [x[0:2] for x in pixels_intensity_labels if x[-1] == False]

    plt.title('Mean pixel intensity on red and green channels')
    plt.scatter([x[0] for x in pixel_negative_labels],[x[1] for x in pixel_negative_labels], color='red', label='Negative Controls',alpha=0.5,s=1)
    plt.scatter([x[0] for x in pixel_positive_labels],[x[1] for x in pixel_positive_labels], color='blue', label='Positive Controls',alpha=0.5,s=1)
    plt.legend(loc='upper right')
    plt.xlabel('red channel',size=15)
    plt.ylabel('green channel',size=15)

def multiply_with_overflow(image, factor):
    m_imageR = cv2.multiply(image[:,:,0], factor[0]) 
    m_image = np.zeros_like(image)
    m_imageR[m_imageR > 1] = 1
    
    m_imageG = cv2.multiply(image[:,:,1], factor[1])
    m_imageG[m_imageG > 1] = 1

    m_imageB = cv2.multiply(image[:,:,2], factor[2])
    m_imageB[m_imageB > 1] = 1

    m_image[:,:,0] = m_imageR
    m_image[:,:,1] = m_imageG
    m_image[:,:,2] = m_imageB
 
    return m_image

def show_mosaic(imgs,figsize):
    h = int(math.sqrt(int(imgs.shape[0])))
    c = int(math.sqrt(int(imgs.shape[0])))

    f,a = plt.subplots(h,c, figsize=figsize)
    
    for i in range(h):
        for j in range(c):
            a[i][j].imshow(np.squeeze(imgs)[h*i + j,:,:], cmap='Greys', interpolation='none')
            a[i][j].axis('off')


def custom_loss(y_true, y_pred):
    red_error = K.mean(K.square(y_pred[:,:,:,0] - y_true[:,:,:,0]), axis=-1)
    green_error = K.mean(K.square(y_pred[:,:,:,1] - y_true[:,:,:,1]), axis=-1)
    return red_error + (10 * green_error)

def add_blue(image):
    return np.stack([image[:,:,0], image[:,:,1], np.zeros_like(image[:,:,1])], axis=2)

def generate_autoencoder1():
    batch_size = 32
    input_img = Input(batch_shape=(batch_size,200, 200, 2))

    e1 = Convolution2D(16, 10, 10, activation='relu', border_mode='same')
    e2 = MaxPooling2D((5,5), border_mode='same')
    e3 = Convolution2D(8, 5, 5, activation='relu', border_mode='same')
    e4 = MaxPooling2D((2,2), border_mode='same')
    e5 = Convolution2D(2, 5, 5, activation='relu', border_mode='same')

    x = e1(input_img)
    x = e2(x)
    x = e3(x)
    x = e4(x)
    x = e5(x)

    encoded = x

    d1 = Deconvolution2D(8, 5, 5, output_shape=(batch_size, 20, 20, 8) , activation='relu', border_mode='same', subsample=(1,1))
    d2 = UpSampling2D((2,2))
    d3 = Deconvolution2D(16, 5, 5, output_shape=(batch_size,40, 40, 16),activation='relu', border_mode='same', subsample=(1,1))
    d4 = UpSampling2D((5,5))
    d5 = Deconvolution2D(2, 10, 10, output_shape=(batch_size, 200, 200, 2), activation='relu', border_mode='same', subsample=(1,1))

    x = d1(x)
    x = d2(x)
    x = d3(x)
    x = d4(x)
    x = d5(x)


    decoder_input = Input(shape=(20,20,2))
    d = d1(decoder_input)
    d = d2(d)
    d = d3(d)
    d = d4(d)
    d = d5(d)
    decoder = Model(decoder_input, d)

    # encoder
    encoder_input = input_img
    e = e1(encoder_input)
    e = e2(e)
    e = e3(e)
    e = e4(e)
    e = e5(e)
    encoder = Model(input_img, e)

    autoencoder = Model(input_img, x)
    rmsprop = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
    autoencoder.compile(optimizer=rmsprop, loss=custom_loss)
    return autoencoder

def generate_convolutional_model():
    nb_classes = 2
    nb_epoch = 50
    img_rows, img_cols = 400, 400

    batch_size = 100
    model = Sequential()

    model.add(Convolution2D(16, 20, 20, border_mode='same',
                            input_shape=(200,200,2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(16, 10, 10))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # model.add(Convolution2D(64, 5, 5, border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(64, 5, 5))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
    return model

def generate_maximal_activations(model,layer_index):
    
    layer = model.layers[layer_index]
    img_shape = [1] + layer.get_input_at(0).get_shape().as_list()[1:]
    nb_filters = layer.get_output_at(0).get_shape().as_list()[-1]

    img_list = []
    for f in range(nb_filters):
        input_img = layer.get_input_at(0)
        layer_output = model.layers[layer_index].get_output_at(0)
        k = int(math.sqrt(nb_filters))
        loss = K.mean(layer_output[:, :, :, f])

        # compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # we start from a gray image with some noise
        input_img_data = np.random.random(img_shape)
        # run gradient ascent for 20 steps
        for i in range(50):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * 10
        
        img_list.append(np.squeeze(input_img_data)[:,:,0])
    return img_list

def load_data(n):
    data = pickle.load(open('intermediate/training_data_{}.py'.format(n),'rb'))
    images, labels = np.array(data[0]), np.array(data[1])
    images = np.array([image[100:300,100:300,:] for image in images])
    positive_images = np.array(images[labels == True])
    positive_labels = np.array(labels[labels == True])
    negative_images = np.array(images[labels == False])
    negative_labels = np.array(labels[labels == False])

    idx = np.random.choice(range(len(negative_images)), len(positive_images))

    equalimages = np.vstack([positive_images, negative_images[idx]])
    equallabels = np.hstack([positive_labels, negative_labels[idx]])

    nb_classes = 2
    X_train, X_test, y_train, y_test = train_test_split(equalimages,equallabels, test_size=0.2)

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, X_test, y_train, y_test, Y_train, Y_test


def generate_autoencoder2(inner_dim, batch_size):

    sys.path = ['./keras'] + sys.path
    from keras.layers import Input, Dense, Convolution2D, Deconvolution2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
    from keras.models import Model
    from keras.optimizers import RMSprop
    from keras.callbacks import TensorBoard
    from keras import backend as K
    from IPython.display import SVG
    from keras.utils.visualize_util import model_to_dot

    
    

    def custom_loss(y_true, y_pred):
        red_error = K.mean(K.square(y_pred[:,:,:,0] - y_true[:,:,:,0]), axis=-1)
        green_error = K.mean(K.square(y_pred[:,:,:,1] - y_true[:,:,:,1]), axis=-1)
        return red_error + (10 * green_error)

    def add_blue(image):
        return np.stack([image[:,:,0], image[:,:,1], np.zeros_like(image[:,:,1])], axis=2)

    ## Layers used
    i1 = Input(batch_shape=(batch_size,200,200,2))
    c1 = Convolution2D(16, 10, 10, activation='relu', border_mode='same', input_shape=(200,200,2))
    p1 = MaxPooling2D((5,5), border_mode='same')
    c2 = Convolution2D(8, 5, 5, activation='relu', border_mode='same')
    p2 = MaxPooling2D((2,2), border_mode='same')
    c3 = Convolution2D(2, 5, 5, activation='relu', border_mode='same')
    f = Flatten()
    d1 = Dense(inner_dim)

    d2 = Dense(800)
    uf = Reshape((20,20,2), input_shape=(batch_size,50))

    dc1 = Deconvolution2D(8, 5, 5, output_shape=(batch_size,20, 20, 8), input_shape=(None,20, 20, 2) , activation='relu', border_mode='same')
    up1 = UpSampling2D((2,2))
    dc2 = Deconvolution2D(16, 5, 5, output_shape=(batch_size,40, 40, 16), input_shape=(None,40, 40, 8) ,activation='relu', border_mode='same')
    up2 = UpSampling2D((5,5))
    dc3 = Deconvolution2D(2, 10, 10, output_shape=(batch_size, 200, 200, 2), input_shape=(None,200, 200, 16), activation='relu', border_mode='same')


    model = Sequential()


    model.add(c1)
    model.add(p1)
    model.add(c2)
    model.add(p2)
    model.add(c3)
    model.add(f)
    model.add(d1)
    # Model is currently encoded
    encoded = model

    model.add(d2)    
    model.add(uf)
    model.add(dc1)
    model.add(up1)
    model.add(dc2)
    model.add(up2)
    model.add(dc3)


    # encoder
    encoder_input = Input(batch_shape=(32,200,200,2))
    e = c1(encoder_input)
    e = p1(e)
    e = c2(e)
    e = p2(e)
    e = c3(e)
    e = f(e)
    e = d1(e)
    encoder = Model(encoder_input, e)

    # decoder
    decoder_input = Input(shape=(inner_dim,))
    d = d2(decoder_input)
    d = uf(d)
    d = dc1(d)
    d = up1(d)
    d = dc2(d)
    d = up2(d)
    d = dc3(d)
    decoder = Model(decoder_input, d)



    rmsprop = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=rmsprop, loss=custom_loss)
    return model