import cv2
from simple_facerec import SimpleFacerec
from datetime import datetime

# Initialize face recognition
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Open webcam
cap = cv2.VideoCapture(0)  # Assuming your webcam index is 1, change it if necessary

# Initialize activity log
#activity_log = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for better usability
    frame = cv2.flip(frame, 1)

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        if name != "Unknown":
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 4)
            # Append the recognized name and timestamp to the activity log
            #activity_log.append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name))

    # Display the frame
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
