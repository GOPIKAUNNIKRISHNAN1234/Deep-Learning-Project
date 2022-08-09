# -*- coding: utf-8 -*-
import cv2
import os
from matplotlib import pyplot as plt
from scipy import ndimage
from matplotlib import cm as cm
from flask import *  
from keras.models import load_model
from app import app
import numpy as np
from tensorflow.keras.preprocessing import image
UPLOAD_FOLDER = 'static/uploads/'
app = Flask(__name__)  
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
letters = []
labels=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90']
CATEGORIES=['്','ാ','ി','ീ','ു','ൂ','ൄ','ര','േ','ൗ','o','അ','ആ','ഇ','ഉ','ഋ','എ','ഏ','ഒ','ക','ഖ','ഗ','ഘ','ങ','ച','ഛ','ജ','ഝ','ഞ','ട','ഠ','ഡ','ഢ','ണ','ത','ഥ','ദ','ധ','ന','പ','ഫ','ബ','ഭ','മ','യ','ര','ല','വ','ശ','ഷ','സ','ഹ','റ','ള','ഴ','ൺ','ൻ','ർ','ൽ','ൾ','ക്ക','ക്ഷ','ങ്ക','ങ്ങ','ച്ച','ഞ്ച','ഞ്ഞ','ട്ട','ണ്ട','ണ്ണ','ത്ത','ദ്ധ','ന്ത','ന്ദ','ന്ന','പ്പ','മ്പ','മ്മ','യ്യ','ല്ല','ള്ള','വ്വ','ൃ','്ര',' ്വ','bba','nma','nma','stha','jnja']
model = load_model('model.h5')
 
@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")  
 
@app.route('/success', methods = ['POST'])  
def success(): 
   if request.method == 'POST': 
        f = request.files['file']	  
        path=f.filename
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],path))
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'],path)
        def prediction(model,img_new):
              x =  cv2.resize(img_new,(80,80))
              cv2.imwrite("data.jpg",x)
              x_img = cv2.imread("data.jpg")
              x = np.expand_dims(x_img, axis=0)
              images = np.vstack([x]) 
              classes = model.predict(images, batch_size=10)
              pred_name=CATEGORIES[np.argmax(classes)]
              print(pred_name)
              
              return pred_name
            

        def _edge_detect(im):
            return np.max(np.array([_sobel_detect(im[:,:, 0]),_sobel_detect(im[:,:, 1]),_sobel_detect(im[:,:, 2])]), axis=0)


        def _sobel_detect(channel): # sobel edge detection
            sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
            sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
            sobel = np.hypot(sobelX, sobelY)
            sobel[sobel > 255] = 255
            return np.uint8(sobel)

        def sort_words(boxes):
            for i in range(1, len(boxes)):
                key = boxes[i]
                j = i - 1
                while(j >= 0 and key[2] < boxes[j][2]):
                    boxes[j+1] = boxes[j]
                    j -= 1
                boxes[j+1] = key
            return boxes

        def sort_boxes(boxes):
            lines = []
            new_lines = []
            tmp_box = boxes[0]
            lines.append(tmp_box)
            for box in boxes[1:]:
                if((box[0] + (box[1] - box[0])/2) < tmp_box[1]):
                    lines.append(box)
                    tmp_box = box
                else:
                    new_lines.append(sort_words(lines))
                    lines = []
                    tmp_box = box
                    lines.append(box)
            new_lines.append(sort_words(lines))
            return(new_lines)

        def sort_labels(label_boxes):
            for i in range(1, len(label_boxes)):
                key = label_boxes[i]
                j = i - 1
                while(j >= 0 and key[1][2] < label_boxes[j][1][2]):
                    label_boxes[j+1] = label_boxes[j]
                    j -= 1
                label_boxes[j+1] = key
            return label_boxes

        def scale_boxes(new_img, old_img, coords):
            coords[0] = int((coords[0]) * new_img.shape[0] / old_img.shape[0]); # new top left
            coords[1] = int((coords[1]) * new_img.shape[1] / old_img.shape[1] ); # new bottom left
            coords[2] = int((coords[2] + 1) * new_img.shape[0] / old_img.shape[0]) - 1; # new top right
            coords[3] = int((coords[3] + 1) * new_img.shape[1] / old_img.shape[1] ) - 1; # new bottom right
            return coords

        def clipping_image(new):
            colsums = np.sum(new, axis=0)
            linessum = np.sum(new, axis=1)
            colsums2 = np.nonzero(0-colsums)
            linessum2 = np.nonzero(0-linessum)

            x = linessum2[0][0] # top left
            xh = linessum2[0][linessum2[0].shape[0]-1] # bottom left
            y = colsums2[0][0] # top right
            yw = colsums2[0][colsums2[0].shape[0]-1] # bottom right

            imgcrop = new[x:xh, y:yw] # crop the image

            return imgcrop, [x, xh, y, yw]


        def padding_resizing_image(img):
            img = cv2.copyMakeBorder(img, 2, 2, 0, 0, cv2.BORDER_CONSTANT) # add 2px padding to image
            try:
                img = cv2.resize(np.uint8(img), (80,80)) # resize the image
            except:
                return img
            finally:
                return img

        def segmentation(img, sequence, origimg=None, wordNo=None):
            if(sequence == "word"): # resize to find the words
                width = 940
                height = int(img.shape[0] * (width / img.shape[1]))
                sigma = 18
            elif(sequence == "character"): # resize to find the characters
                width = img.shape[1] # 1280
                height = img.shape[0] # int(img.shape[0] * (width / img.shape[1]))
                sigma = 0

            img = cv2.resize(img, (width, height))
            blurred = cv2.GaussianBlur(img, (5, 5), sigma) # apply gaussian blur

            if(sequence == "word"):
                blurred = _edge_detect(blurred) # edge detect in blurred image (words)
                ret, img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Otsu's thresholding with Binary
                img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8)) # Morphological processing - Black&White
              
            elif(sequence == "character"):
                ret, img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) # Otsu's thresholding with Binary Inverted
            

            num_labels, labels_im = cv2.connectedComponents(img) # find the connected components

            if(sequence == "word"):
                boxes = [] # for storing the coordinates of the bounding boxes
                for i in range(1, num_labels):
                    new, nr_objects = ndimage.label(labels_im == i) # select the images with label
                    new, new_coord = clipping_image(new) # clipping the image to the edges
                    if(not(new.shape[0] < 10 or new.shape[1] < 10)):
                        boxes.append(new_coord)

            if(sequence == "character"):
                boxes = []
                label_box = []
                for i in range(1, num_labels):
                    new, nr_objects = ndimage.label(labels_im == i) # select the images with label
                    new, new_coord = clipping_image(new) # clipping the image to the edges
                    if(not(new.shape[0] < 10 or new.shape[1] < 10)):
                        label_box.append([i, new_coord])
                label_box = sort_labels(label_box) # sort the words
                chNo = 0 
                for box in label_box:
                    ch_img, nr_objects = ndimage.label(labels_im == box[0])
                    ch_img, new_coord = clipping_image(ch_img)
                    cropped_image = padding_resizing_image(ch_img)
                    try:
            
                        plt.imshow(cropped_image, cmap=cm.gray)
                      
                        letters.append(prediction(model,cropped_image))
                        
                        
                    except Exception as e:
                        print("passed")
                        print(e)
                        pass
                    finally:
                        pass
                    chNo += 1
            return img, boxes

        
        img = cv2.cvtColor(cv2.imread(full_filename),cv2.COLOR_BGR2RGB)

        kernel = np.ones((5,5),np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) # opening image to remove minor noise
        new_img, boxes = segmentation(img, "word") # line segmentation
        boxes.sort()
        boxes = sort_boxes(boxes)
        scaled_boxes = [] # coordinates values scaled to img
        for box in boxes:
            for box in box:
                box = scale_boxes(img, new_img, box)
                scaled_boxes.append(box)
        wordNo = 0
        for scaled_box in scaled_boxes:
            img_gray = cv2.cvtColor(img[scaled_box[0]:scaled_box[1], scaled_box[2]:scaled_box[3]], cv2.COLOR_BGR2GRAY)
            img_new, _ = segmentation(img_gray, "character", None, wordNo)
            print(wordNo)
            wordNo += 1
        word ="".join(letters)
        print(word)
        
   return render_template("file_upload_form.html", name = word,user_image =full_filename)
@app.route('/display/<path>')
def display_image(path):
	#print('display_image filename: ' + path)
      return redirect(url_for('static', filename='uploads/' + path), code=301) 
  
if __name__ == '__main__':  
    app.run(debug = True) 