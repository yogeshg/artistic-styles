from PIL import Image
import numpy

def preprocess_image(paths,resize=True):
    # if single given not list of strings then convert to list of strings
    if isinstance(paths, basestring):
        paths = [paths]
    if resize == False:
        assert len(paths)==1
    images = []
    for path in paths:
        image = Image.open(path,'r').convert('RGB')
        if resize:
            w,h = image.size
            # resize so smallest dimenison = 256 (preserve aspect ratio)
            if resize:
                if h<w:
                    image = image.resize((w*256/h,256),resample=Image.BILINEAR)
                else:
                    image = image.resize((256,h*256/w),resample=Image.BILINEAR)
                # crop the images to 224x224
                #get the new shape of the image
                w1,h1 = image.size
                #get the bounds of the box
                right = w1//2 + 112
                left =w1//2 - 112
                top = h1//2 - 112
                bottom = h1//2 + 112
                image = image.crop((left,top,right,bottom))

        im = numpy.asarray(image)
        imcopy = numpy.zeros(im.shape)
        imcopy[:,:,0] = im[:, :, 0] - 103.939
        imcopy[:,:,1] = im[:, :, 1] - 116.779
        imcopy[:,:,2] = im[:, :, 2] - 123.68
        #RGB -> BGR
        imcopy = imcopy[:, :, ::-1]
        #put channels first
        imcopy = numpy.rollaxis(imcopy,2,0)
        #add dimension to make it a 4d image (for theano tensor)
        imcopy = numpy.expand_dims(imcopy,axis=0)
        #store it in images array
        if len(images)==0:
            images = imcopy
        else:
            images = numpy.append(images,imcopy,axis=0)
    return images

def deprocess_image(image_array):
    # put channels last
    assert image_array.ndim==4
    image_array = numpy.rollaxis(image_array,1,4)
    #BGR -> RGB
    image_array = image_array[:,:,:,::-1]
    #add mean channel
    image_array_copy = numpy.zeros(image_array.shape)
    image_array_copy[:,:,:,0] = image_array[:,:,:,0] + 103.939
    image_array_copy[:,:,:,1] = image_array[:,:,:,1] + 116.779
    image_array_copy[:,:,:,2] = image_array[:,:,:,2] + 123.68
    #convert to int between 0 and 254
    image_array_copy = numpy.clip(image_array_copy, 0, 255).astype('uint8')
    return image_array_copy

def np2pil(image_array):
    assert image_array.ndim==3
    image = Image.fromarray(image_array)
    return image