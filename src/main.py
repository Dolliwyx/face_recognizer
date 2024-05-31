import face_recognition
import cv2
import numpy as np
import time
import os

img_size = (250, 250)
folder = "captured"
directory = f"{folder}/test"
subject_id = None
counter = 1


def main():
    global subject_id, directory, counter

    vid_capture = cv2.VideoCapture(0)

    process_this_frame = True

    optional_directory = input("Enter directory: ")
    optional_subject_id = input("Enter the subject ID: ")

    if optional_subject_id:
        subject_id = optional_subject_id
    else:
        raise ValueError("Subject ID is required")

    if optional_directory:
        directory = f"{folder}/{optional_directory}"

        # create directory if it doesn't exist yet
        if not os.path.exists(directory):
            os.makedirs(directory)

    while True:
        _, frame = vid_capture.read()

        if process_this_frame:
            # Resize the frame of video to 1/4 size for faster face recognition processing
            sm_frame: np.ndarray = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert color from BGR to RGB
            rgb_frame: np.ndarray = sm_frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)

        process_this_frame = not process_this_frame

        for top, right, bottom, left in face_locations:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            if cv2.waitKey(1) == 13:  # 13 is the Enter key
                capture_face(frame, (top, right, bottom, left))

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # make the video window smaller
        flipped_frame = cv2.flip(frame, 1)
        cv2.imshow(
            "detector ng mukha", cv2.resize(flipped_frame, (0, 0), fx=0.5, fy=0.5)
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vid_capture.release()
    cv2.destroyAllWindows()


def capture_face(frame: np.ndarray, coordinates: tuple[int, int, int, int]) -> None:
    """
    Captures the face in the frame and aligns the eyes

    Parameters:
    frame (numpy.ndarray): frame to capture the face from
    coordinates (Tuple[int, int, int, int]): coordinates of the face in the frame
    """
    global subject_id, directory, counter

    # Extract the face region from the frame
    face: np.ndarray = frame[
        coordinates[0] : coordinates[2], coordinates[3] : coordinates[1]
    ]

    # Detect facial landmarks (including eyes)
    landmarks = face_recognition.face_landmarks(face)

    # Assuming one face is detected
    if len(landmarks) > 0:
        # Get the locations of the left and right eyes
        left_eye = landmarks[0]["left_eye"]
        right_eye = landmarks[0]["right_eye"]

        # Calculate the center of the eyes
        left_eye_center = np.mean(left_eye, axis=0).astype("int")
        right_eye_center = np.mean(right_eye, axis=0).astype("int")

        # Calculate the angle between the eyes
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # Rotate the frame to align the face
        rotated_frame = cv2.warpAffine(
            frame,
            cv2.getRotationMatrix2D(
                (frame.shape[1] // 2, frame.shape[0] // 2), angle, 1.0
            ),
            (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_LINEAR,
        )

        # Extract the rotated face region from the rotated frame
        rotated_face = rotated_frame[
            coordinates[0] : coordinates[2], coordinates[3] : coordinates[1]
        ]

        file_name = f"{subject_id }_{time.time()}.jpg"
        path = f"{directory}/{file_name}"

        cv2.imwrite(path, cv2.resize(rotated_face, img_size))
        print(f"Saved at {path} - {counter}")
        counter += 1

    else:
        print("No landmarks found for the face, skipping...")

main()