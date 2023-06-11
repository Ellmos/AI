from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import random
import sys
import numpy as np


BLACK = (255, 255, 255, 255)
WHITE = (0, 0, 0, 255)
DISPLACEMENT = 2

width = 28
height = 28

directory = os.path.dirname(os.path.abspath(__file__)) + '/ownDataSet/'
fontDirectory = directory + "fonts/"

fonts = []
for f in os.listdir(fontDirectory):
    fonts.append(fontDirectory + f)

lenFont = len(fonts)

def create_image(iteration, isTrainDataSet):
    if isTrainDataSet:
        imageFile = open(directory + "ownTrainImages.bytes", "wb")
        labelFile = open(directory + "ownTrainLabels.bytes", "wb")
    else:
        imageFile = open(directory + "ownTestImages.bytes", "wb")
        labelFile = open(directory + "ownTestLabels.bytes", "wb")


    imageFile.write(iteration.to_bytes(8, "little"))
    labelFile.write(iteration.to_bytes(8, "little"))

    for i in range(iteration):
        num = random.randrange(0, 10)
        rotate = random.randrange(-10, 10)

        img = Image.new(mode="RGB", size=(width, height), color=WHITE)
        img = img.rotate(rotate, fillcolor=WHITE)

        draw = ImageDraw.Draw(img)

        for j in range(random.randrange(0, 2)):
            x = random.randrange(0, width)
            y = random.randrange(0, height)
            x_size = random.randrange(1, 2)
            y_size = random.randrange(1, 2)
            draw.rectangle((x, y, x+x_size, y+y_size), fill=BLACK)
        
        imgX = img.filter(ImageFilter.GaussianBlur(0.1))
        drawX = ImageDraw.Draw(imgX)

        d_x = random.randrange(-DISPLACEMENT, DISPLACEMENT)
        d_y = random.randrange(-DISPLACEMENT, DISPLACEMENT)

        fontname = fonts[random.randrange(0, lenFont)]
        font = ImageFont.truetype(fontname, random.randrange(10, 24))
        drawX.text((width/2+d_x, height/2+d_y), str(num), font=font, anchor="mm", fill=BLACK)

        imgX = imgX.rotate(-rotate, fillcolor=WHITE)

        blur = imgX.filter(ImageFilter.GaussianBlur(0.1))


        img = blur.convert('L')
        # imageFilename = hashlib.sha256(str(random.getrandbits(256)).encode('utf-8')).hexdigest()[:8]
        # img.save("./output/"+str(num)+"__"+imageFilename+".bmp", format="BMP")



        img = np.array(img.getdata())
        m = img.max()
        if m != 0:
            img = img * (1.0/m)
        else:
            img = img.astype(np.float64)
        
        imageFile.write(img.tobytes())
        labelFile.write(num.to_bytes(1, "little"))

        if i % 200 == 0:
            print(i)

    imageFile.close()
    labelFile.close()


create_image(int(sys.argv[1]), True)
create_image(int(sys.argv[2]), False)
