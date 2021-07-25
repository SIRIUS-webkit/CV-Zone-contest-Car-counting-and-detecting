import cv2

cap = cv2.VideoCapture('DRONE-SURVEILLANCE-CONTEST-VIDEO.mp4')

ret, frame1 = cap.read()
frame1 = cv2.resize(frame1, (640, 480), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
ret, frame2 = cap.read()
frame2 = cv2.resize(frame2, (640, 480), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
while True:
    diff = cv2.absdiff(frame1, frame2)
    grey = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5,5), 0)
    _,thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilatted = cv2.dilate(thresh, None, iterations=3)
    contours, _= cv2.findContours(dilatted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        if cv2.contourArea(cnt) < 900:
            continue
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("output", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()
    frame2 = cv2.resize(frame2, (640, 480), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
cap.release()
cv2.destroyAllWindows()

