import argparse
import srt
import pytesseract
import cv2
from imutils.object_detection import non_max_suppression
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from fuzzywuzzy import fuzz
import time


def decode_predictions(scores, geometry):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < 0.95:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return rects, confidences


def main(args):
    start_program = time.time()

    videoFilePath = args.video
    modelFilePath = args.model
    pytesseractLanguage = args.language

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ]

    net = cv2.dnn.readNet(modelFilePath)
    stream = cv2.VideoCapture(videoFilePath)
    video_fps = stream.get(cv2.CAP_PROP_FPS)
    (newW, newH) = (160, 160)
    (rW, rH) = (None, None)
    frame_count = 0
    total_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_frames, unit='frames')
    subtitles = []

    previous_text = None
    current_subtitle = None

    while True:
        frame_count += 1
        ret, frame = stream.read()
        if not ret:
            break

        if frame_count % args.frame_rate != 0:
            continue

        orig = frame.copy()
        (origH, origW) = frame.shape[:2]
        rW = origW / float(newW)
        rH = origH / float(newH)
        frame = cv2.resize(frame, (newW, newH))

        blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        (rects, confidences) = decode_predictions(scores, geometry)
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        if len(boxes) != 0:
            text = pytesseract.image_to_string(orig, config=f"-l {pytesseractLanguage} --oem 1 --psm 3")

            start_time_ms = stream.get(cv2.CAP_PROP_POS_MSEC)
            end_time_ms = start_time_ms + ((args.frame_rate / video_fps) * 1000)

            if previous_text is None or fuzz.ratio(previous_text, text) < 90:
                subtitle = srt.Subtitle(index=frame_count,
                                        start=timedelta(milliseconds=start_time_ms),
                                        end=timedelta(milliseconds=end_time_ms),
                                        content=text)
                subtitles.append(subtitle)
                current_subtitle = subtitle
            else:
                current_subtitle.end = timedelta(milliseconds=end_time_ms)

            previous_text = text

        for (startX, startY, endX, endY) in boxes:
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 1)

        if args.preview:
            cv2.imshow("Preview", orig)

        progress_bar.update(args.frame_rate)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    progress_bar.close()
    stream.release()
    cv2.destroyAllWindows()

    output_srt_filename = videoFilePath.rsplit('.', 1)[0] + "_" + pytesseractLanguage + "_" + datetime.now().strftime(
        "%Y-%m-%d-%H-%M") + ".srt"
    print(f"Preparing to write to file: {output_srt_filename}")

    try:
        with open(output_srt_filename, 'w', encoding='utf-8') as f:
            f.write(srt.compose(subtitles))
        print("File written successfully")
        end_program = time.time()
        print(f"Total processing time: {end_program - start_program} seconds")
    except Exception as e:
        print(f"Error while writing to file: {e}")
        end_program = time.time()
        print(f"Total processing time: {end_program - start_program} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract text from video using OCR and generate SRT file')
    parser.add_argument('-v', '--video', help='Path to the video file', required=True)
    parser.add_argument('-m', '--model', help='Path to the pre-trained EAST text detector model', required=True)
    parser.add_argument('-l', '--language', help='Language model for Pytesseract', default='eng')
    parser.add_argument('-f', '--frame_rate', help='Number of frames to skip for processing', type=int, default=10)
    parser.add_argument('-p', '--preview', help='Enable preview of the video with bounding boxes', action='store_true')
    args = parser.parse_args()

    main(args)
