import face_recognition
import cv2
import numpy as np
import time
from sys import argv

interval = 2
counter = 0
img_size = (250, 250)


def main():
    global counter, interval
    vid_capture = cv2.VideoCapture(2)

    process_this_frame = True

    last_time = time.time()

    while True:
        _, frame = vid_capture.read()

        if process_this_frame:
            # Resize the frame of video to 1/4 size for faster face recognition processing
            sm_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert color from BGR to RGB
            rgb_frame = sm_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_frame)
            # face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        process_this_frame = not process_this_frame

        for top, right, bottom, left in face_locations:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            if time.time() - last_time >= interval and counter < 10:
                last_time = time.time()
                capture_face(frame, (top, right, bottom, left))
                print(f"Saved face{counter}.jpg last {last_time}")
                counter += 1

        # make the video window smaller
        cv2.imshow('detector ng mukha', cv2.resize(
            frame, (0, 0), fx=0.5, fy=0.5))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid_capture.release()
    cv2.destroyAllWindows()


def capture_face(frame: np.ndarray, coordinates: tuple[int, int, int, int]) -> None:
    """
    Captures the face in the frame

    Parameters:
    frame (numpy.ndarray): frame to capture the face from
    coordinates (Tuple[int, int, int, int]): coordinates of the face in the frame
    """
    global counter, interval

    face: np.ndarray = frame[coordinates[0]:coordinates[2], coordinates[3]:coordinates[1]]
    cv2.imwrite(f'face{counter}.jpg', cv2.resize(face, img_size))


def capture_face_from_file(file_name: str) -> None:
    """
    Captures the face from the given file name, if any

    Parameters:
    file_name (str): name of the file to capture the face from

    Returns:
    None
    """
    img = face_recognition.load_image_file(file_name)

    if img is None:
        return print('Could not find the image')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(img)
    if not len(face_locations):
        return print('No faces found')

    timestamp = time.time()

    for top, right, bottom, left in face_locations:
        face = img[top:bottom, left:right]
        cv2.imwrite(f'face-from-img_{timestamp}.jpg',
                    cv2.resize(face, img_size))
        print(f'Found and saved image face-from-img_{timestamp}.jpg')


if not len(argv):
    main()
else:
    print(f'Capturing face from {argv[1]}...')
    capture_face_from_file(argv[1])
