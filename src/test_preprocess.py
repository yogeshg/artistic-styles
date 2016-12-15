from imageProcess import preprocess_image

if __name__ == '__main__':

    print preprocess_image('test_images/thais.JPG').shape,'should be (1,3,224,224)'
    print preprocess_image('test_images/thais.JPG',resize=False).shape,'should be (1,3,3000,4000)'
    print preprocess_image(['test_images/thais.JPG','test_images/bottle.jpg']).shape, 'should be (2,3,244,244)'
