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

    plt.title('Mean pixel intensity on red and blue channels')
    plt.scatter([x[0] for x in pixel_negative_labels],[x[1] for x in pixel_negative_labels], color='red', label='False',alpha=0.5)
    plt.scatter([x[0] for x in pixel_positive_labels],[x[1] for x in pixel_positive_labels], color='green', label='True',alpha=0.5)
    plt.legend(loc='lower right')
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