import face_recognition
import cv2
import numpy as np
import time

interval = 2
counter = 0
img_size = (250, 250)


def main() -> None:
    """
    Main function to capture images from webcam and save them to disk

    Returns:
        None
    """
    counter: int = 0
    interval: float = 2.0
    last_time: float = time.time()
    process_this_frame: bool = True
    vid_capture: cv2.VideoCapture = cv2.VideoCapture(2)

    while True:
        _, frame = vid_capture.read()

        if process_this_frame:
            # Resize the frame of video to 1/4 size for faster face recognition processing
            sm_frame: np.ndarray = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert color from BGR to RGB
            rgb_frame: np.ndarray = sm_frame[:, :, ::-1]

            face_locations: list[tuple[int, int, int, int]
                                 ] = face_recognition.face_locations(rgb_frame)
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


main()
