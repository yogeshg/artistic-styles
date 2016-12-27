from imageProcess import preprocess_image

if __name__ == '__main__':

    print preprocess_image('test_images/thais.JPG')[0].shape,'should be (1,3,224,224)'
    print preprocess_image('test_images/thais.JPG',shape=(122,1045))[0].shape,'should be (1,3,122,1045)'
    print preprocess_image('test_images/thais.JPG',shape=(150,40))[0].shape,'should be (1,3,150,40)'
    print preprocess_image('test_images/thais.JPG',resize=False)[0].shape,'should be (1,3,3000,4000)'
    print preprocess_image(['test_images/thais.JPG','test_images/bottle.jpg'])[0].shape, 'should be (2,3,244,244)'
