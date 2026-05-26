import cv2
import onnxruntime
import numpy as np

INPUT_WIDTH = 512
INPUT_HEIGHT = 384
ROAD_CLASS_ID = 1

segmentation_colors = np.array([
    [0, 0, 0],
    [255, 191, 0],
    [192, 67, 251]
], dtype=np.uint8)


# ======================PREPROCESS======================
def prepare_input(image):
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, (INPUT_WIDTH, INPUT_HEIGHT))

    # Normalize in-place for efficiency
    input_img = input_img / 255.0
    input_img -= np.array([0.485, 0.456, 0.406])
    input_img /= np.array([0.229, 0.224, 0.225])

    input_img = input_img.transpose(2, 0, 1)
    input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

    return input_tensor

def inference(session, input_tensor):
    # Get model inputs/outputs
    model_inputs = session.get_inputs()
    input_names = [model_inputs[i].name for i in range(len(model_inputs))]
    model_outputs = session.get_outputs()
    output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    # Run inference
    outputs = session.run(output_names, {input_names[0]: input_tensor})
    return outputs, output_names


# ====================== DRAW SEG ======================
def draw_seg(seg_map, image, alpha=0.5):
    color_segmap = segmentation_colors[seg_map]
    color_segmap = cv2.resize(color_segmap, (image.shape[1], image.shape[0]))

    combined_img = cv2.addWeighted(image, alpha, color_segmap, (1 - alpha), 0)
    return combined_img


# ====================== CENTER LINE (IMPROVED) ======================
def compute_center_line(seg_map, frame):
    H, W = frame.shape[:2]
    seg_full = cv2.resize(
        seg_map.astype(np.uint8),
        (W, H),
        interpolation=cv2.INTER_NEAREST
    )
    road_mask = (seg_full == ROAD_CLASS_ID)
    center_points = []

    for y in range(int(H * 0.6), int(H * 0.85)):
        xs = np.where(road_mask[y])[0]
        if len(xs) > W * 0.05:
            # KEY: percentile instead of min/max
            x_left = np.percentile(xs, 10)
            x_right = np.percentile(xs, 90)
            center_x = int((x_left + x_right) / 2)
            center_points.append((center_x, y))

    # smoothing (moving average) 
    smoothed = []
    if len(center_points) > 0:
        pts = np.array(center_points)
        window = 5

        for i in range(len(pts)):
            start = max(0, i - window)
            end = min(len(pts), i + window)
            avg_x = int(np.mean(pts[start:end, 0]))
            smoothed.append((avg_x, pts[i, 1]))

    return smoothed


# ====================== DRAW CENTER LINE ======================
def draw_center_line(frame, points):
    if len(points) > 1:
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], False, (0, 255, 0), 10)
    return frame


# ====================== MAIN ======================
#cap = cv2.VideoCapture("test_videos/4k1.mp4")
cap = cv2.VideoCapture(1)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

session = onnxruntime.InferenceSession("model_data/hybridnets_384x512.onnx")

"""out = cv2.VideoWriter(
    'outputs/4k1_3.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    20,
    (1280, 720)
)"""

cv2.namedWindow("Road Detections", cv2.WINDOW_NORMAL)

frame_count = 0
skip_frames = 2

while cap.isOpened():
    if cv2.waitKey(1) == ord('q'):
        break
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % skip_frames == 0:
        # ===== Inference =====
        input_tensor = prepare_input(frame)
        outputs, output_names = inference(session, input_tensor)
        out_seg = outputs[output_names.index("segmentation")]
        seg_map = np.squeeze(np.argmax(out_seg, axis=1))

        # ===== Draw seg =====
        vis = draw_seg(seg_map, frame)

        # ===== Center line =====
        center_points = compute_center_line(seg_map, frame)
        vis = draw_center_line(vis, center_points)

        # ===== Show =====
        cv2.imshow("Road Detections", vis)
        #vis_out = cv2.resize(vis, (1280, 720))
        #out.write(vis_out)

    frame_count += 1


cap.release()
#out.release()
cv2.destroyAllWindows()
