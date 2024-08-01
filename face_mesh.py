import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
        
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()


















# import cv2
# import mediapipe as mp
# import time

# cap = cv2.VideoCapture(0)
# pTime = 0
# cTime = 0

# mpFaceMesh = mp.solutions.face_mesh
# mpDraw = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# faceMesh = mpFaceMesh.FaceMesh()
# drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

# while True:
#     success, img = cap.read()
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = faceMesh.process(imgRGB)
#     # if results.multi_face_landmarks:
#     for faceLms in results.multi_face_landmarks:
#         mpDraw.draw_landmarks(
#             image=img,
#             landmark_list=faceLms,
#             connections=mpFaceMesh.FACEMESH_TESSELATION,
#             landmark_drawing_spec=None,
#             connection_drawing_spec=mp_drawing_styles
#             .get_default_face_mesh_tesselation_style())
#         mpDraw.draw_landmarks(
#             image=img,
#             landmark_list=faceLms,
#             connections=mpFaceMesh.FACEMESH_CONTOURS,
#             landmark_drawing_spec=None,
#             connection_drawing_spec=mp_drawing_styles
#             .get_default_face_mesh_contours_style())
#         mpDraw.draw_landmarks(
#             image=img,
#             landmark_list=faceLms,
#             connections=mpFaceMesh.FACEMESH_IRISES,
#             landmark_drawing_spec=None,
#             connection_drawing_spec=mp_drawing_styles
#             .get_default_face_mesh_iris_connections_style())

#     cTime = time.time()
#     fps = 1/(cTime - pTime)
#     pTime = cTime
#     cv2.putText(img , str(round(fps)) , (20,70) , cv2.FONT_HERSHEY_COMPLEX , 3 , (0,255,0) , 3)

#     cv2.imshow("Image",img)
#     cv2.waitKey(1)


# while True:
#     success , img = cap.read()
#     # print(success)
#     imgRGB = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)
#     results = faceMesh.process(imgRGB)
#     if results.multi_face_landmarks:
#         for faceLMS in results.multi_face_landmarks:
#             mpDraw.draw_landmarks(img , faceLMS , mpFaceMesh.FACE_CONNECTIONS, drawSpec, drawSpec)
#             for lm in faceLMS.Landmark:
#                 print(lm)
#                 # ih, iw, ic = img.shape
#                 # x, y = int(lm.x*iw), int(lm.y*ih)
#                 # print(x, y)
