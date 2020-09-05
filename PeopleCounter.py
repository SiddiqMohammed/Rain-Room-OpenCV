import numpy as np
import cv2 as cv

try:
    log = open('log.txt',"w")
except:
    print( "No se puede abrir el archivo log")

#Contadores de entrada y salida
cnt_up   = 0
cnt_down = 0

#Fuente de video
#cap = cv.VideoCapture(0)
cap = cv.VideoCapture('Test Files/videos/TestVideo.avi')


#Imprime las propiedades de captura a consola
for i in range(19):
    print( i, cap.get(i))


if cap.isOpened():
    h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)  # float
    w = cap.get(cv.CAP_PROP_FRAME_WIDTH)  # float

#Calculate Gx and Gy for grid lines
gX = int(w/3)
gY = int(h/3)

gx1 = gX
gy1 = gY
gx2 = gX*2
gy2 = gY*2
gx3 = int(w)
gy3 = int(h)


frameArea = h*w
areaTH = frameArea/250
print( 'Area Threshold', areaTH)

#Lineas de entrada/salida
line_up = int(2*(h/5))
line_down   = int(3*(h/5))


up_limit =   int(1*(h/5))
down_limit = int(4*(h/5))


#Substractor de fondo
fgbg = cv.createBackgroundSubtractorMOG2(detectShadows = True)

#Elementos estructurantes para filtros morfoogicos
kernelOp = np.ones((3,3),np.uint8)
kernelOp2 = np.ones((5,5),np.uint8)
kernelCl = np.ones((11,11),np.uint8)

#Variables
font = cv.FONT_HERSHEY_SIMPLEX
persons = []
max_p_age = 5
pid = 1


color1 = (255, 255, 255)
color2 = (0, 0, 255)
cg1 = color1
cg2 = color1
cg3 = color1
cg4 = color1
cg5 = color1
cg6 = color1
cg7 = color1
cg8 = color1
cg9 = color1


while(cap.isOpened()):
    #Lee una imagen de la fuente de video
    ret, frame = cap.read()
    
    #Drawing the grid
    # cv.line(frame, (0, gy1), (gx3, gy1), (150, 0, 200), 2)
    # cv.line(frame, (0, gy2), (gx3, gy2), (150, 0, 200), 2)
    # cv.line(frame, (gx1, 0), (gx1, gy3), (150, 0, 200), 2)
    # cv.line(frame, (gx2, 0), (gx2, gy3), (150, 0, 200), 2)
    
    # Row 1
    cv.rectangle(frame, (0, 0), (gx1, gy1), cg1, 2)
    cv.rectangle(frame, (gx1, 0), (gx2, gy1), cg2, 2)
    cv.rectangle(frame, (gx2, 0), (gx3, gy1), cg3, 2)

    # Row 2
    cv.rectangle(frame, (0, gy1), (gx1, gy2), cg4, 2)
    cv.rectangle(frame, (gx1, gy1), (gx2, gy2), cg5, 2)
    cv.rectangle(frame, (gx2, gy1), (gx3, gy2), cg6, 2)

    # Row 3
    cv.rectangle(frame, (0, gy2), (gx1, gy3), cg7, 2)
    cv.rectangle(frame, (gx1, gy2), (gx2, gy3), cg8, 2)
    cv.rectangle(frame, (gx2, gy2), (gx3, gy3), cg9, 2)


    for i in persons:
        i.age_one() #age every person one frame
    
    #Aplica substraccion de fondo
    fgmask = fgbg.apply(frame)
    fgmask2 = fgbg.apply(frame)

    #Binariazcion para eliminar sombras (color gris)
    try:
        ret,imBin= cv.threshold(fgmask,200,255,cv.THRESH_BINARY)
        ret,imBin2 = cv.threshold(fgmask2,200,255,cv.THRESH_BINARY)
        #Opening (erode->dilate) para quitar ruido.
        mask = cv.morphologyEx(imBin, cv.MORPH_OPEN, kernelOp)
        mask2 = cv.morphologyEx(imBin2, cv.MORPH_OPEN, kernelOp)
        #Closing (dilate -> erode) para juntar regiones blancas.
        mask =  cv.morphologyEx(mask , cv.MORPH_CLOSE, kernelCl)
        mask2 = cv.morphologyEx(mask2, cv.MORPH_CLOSE, kernelCl)
    except:
        print('EOF')
        print( 'UP:',cnt_up)
        print ('DOWN:',cnt_down)
        break

    
    # RETR_EXTERNAL returns only extreme outer flags. All child contours are left behind.
    contours0, hierarchy = cv.findContours(mask2,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        area = cv.contourArea(cnt)
        if area > areaTH:

            #Falta agregar condiciones para multipersonas, salidas y entradas de pantalla.
            
            M = cv.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            cv.circle(frame,(cx,cy), 5, (0,0,255), -1)
            rectSize = 50
            cv.rectangle(frame,(cx+rectSize,cy+rectSize), (cx-rectSize,cy-rectSize), (0,0,255), 2)

            text = cx, cy
            cv.putText(frame, str(text), (cx,cy), font, 0.5, (255,0,0), 1, cv.LINE_AA)

            # for ccx in range(1, 4):
                # for ccy in range(1, 4):
            if cx > 0 and cx < gx1 and cy > 0 and cy < gy1:
                cg1 = color2
            else:
                cg1 = color1



    str_up = 'UP: '+ str(cnt_up)
    str_down = 'DOWN: '+ str(cnt_down)


    cv.imshow('Frame',frame)
    cv.imshow('Mask',mask)    
    


    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
#END while(cap.isOpened())

log.flush()
log.close()
cap.release()
cv.destroyAllWindows()
