
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
win1 = "ascii"
win2 = "capture"
cv.namedWindow(win1, cv.WINDOW_FREERATIO)

cv.namedWindow(win2, cv.WINDOW_FREERATIO)

W = 400
size = 480, 640, 3
rook_image = np.zeros(size)
cv.rectangle(rook_image, (0, 0), (640, 480), (255, 255, 255), -1, 8)

ascii_chars = " . : - = + * % @ "

ascii_chars2 = "$ @ B % 8 & W M # * o a h k b d p q w m Z O 0 Q L C J U Y X z c v u n x r j f t / | ( ) 1 [ ] ? - _ + ~ < > i ! l I ; : , ^ ` ' .    "

charList = ascii_chars2.split(" ", -1)


every = 7

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()

    if not ret:
        print("cant recieve frame (stream end??) exiting...")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    height, width, p = frame.shape
    
    ascii_frame = np.ones((height, width, 3), dtype=np.uint8) * 255

    for row in range(height):
        for col in range(width):
            if row % every == 0 and col % every == 0:
                pixel = frame[row][col]
                R = pixel[0]
                G = pixel[1]
                B = pixel[2]
                Y = ((0.2126 * R) + (0.7152 * G) + (0.0722 * B))/3.8
                
                current = charList[int(Y)]
                
                cv.putText(ascii_frame, current, (col, row), cv.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1, cv.LINE_AA)

    cv.imshow(win1, ascii_frame)
    cv.imshow(win2, gray)
    # cv.putText(frame, 'ascii', (10, 10), cv.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 2, cv.LINE_AA)
    
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()




