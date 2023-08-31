import argparse
import srt
import json
import pytesseract
import cv2
from imutils.object_detection import non_max_suppression
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm


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
    # Define paths for the video file and the pre-trained model file.
    videoFilePath = args.video
    modelFilePath = args.model

    # Configuring Pytesseract to use the correct language model
    pytesseractLanguage = args.language

    # Configuring Pytesseract to blacklist letters
    pytesseractBlacklist = args.blacklist

    # Layer names from pre-trained EAST text detector model to extract scores and geometry data.
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ]

    # Loading the pre-trained EAST text detector model.
    net = cv2.dnn.readNet(modelFilePath)

    # Open the video file.
    stream = cv2.VideoCapture(videoFilePath)

    # Fetch video fps
    video_fps = stream.get(cv2.CAP_PROP_FPS)

    # Initiate some variables.
    (newW, newH) = (160, 160)
    (rW, rH) = (None, None)
    frame_count = 0
    total_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_frames, unit='frames')

    # Create an empty list to hold the subtitles.
    subtitles = []

    # Create an empty list to hold the JSON data.
    json_output = []

    # Main loop for processing the video frames.
    while True:
        frame_count += 1
        ret, frame = stream.read()
        if not ret:
            break

        # Applying EAST text detector and OCR to every nth frame of the video, where n is defined by args.frame_rate.
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

        # Perform OCR on the original frame, not the resized one.
        if len(boxes) != 0:
            text = pytesseract.image_to_string(orig, config=f"-l {pytesseractLanguage} --oem 1 --psm 3 -c tessedit_char_blacklist={pytesseractBlacklist}")

            # Extract OCR confidence score
            data = pytesseract.image_to_data(orig, config=f"-l {pytesseractLanguage} --oem 1 --psm 3", output_type=pytesseract.Output.DICT)
            confidence_scores = [int(conf) for conf in data['conf'] if str(conf).isdigit()]
            confidence_tesseract = sum(confidence_scores) / len(confidence_scores) if confidence_scores else -1.0

            # Define the timing and length for each subtitle object.
            start_time_ms = stream.get(cv2.CAP_PROP_POS_MSEC)
            end_time_ms = start_time_ms + ((args.frame_rate / video_fps) * 1000)

            # Create a new Subtitle object and add it to the list of subtitles.
            subtitle = srt.Subtitle(index=frame_count,
                                    start=timedelta(milliseconds=start_time_ms),
                                    end=timedelta(milliseconds=end_time_ms),
                                    content=text)
            subtitles.append(subtitle)

            # Convert confidences to float
            confidences = [float(confidence) for confidence in confidences]
            # Calculate average confidence score for detection
            average_detection_confidence = sum(confidences) / len(confidences)

            # Append data to json_output list
            json_output.append({
                'frame_number': frame_count,
                'timecode_ms': start_time_ms,
                'text_detection_confidence': average_detection_confidence,  # The confidence text being detected
                'ocr_text': text,
                'ocr_confidence': confidence_tesseract  # The confidence of the OCR string
            })

        # Draw bounding boxes around text areas on the original frame.
        for (startX, startY, endX, endY) in boxes:
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 1)

        # Display a video preview with bounding boxes if the preview is enabled.
        if args.preview:
            cv2.imshow("Preview", orig)

        # Update progress bar.
        progress_bar.update(args.frame_rate)

        # Exit the loop if 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Close progress bar, release the video file and destroy all windows.
    progress_bar.close()
    stream.release()
    cv2.destroyAllWindows()

    # Define the output filename for the SRT file.
    output_srt_filename = videoFilePath.rsplit('.', 1)[0] + "_" + pytesseractLanguage + "_" + datetime.now().strftime(
        "%Y-%m-%d-%H-%M") + ".srt"
    print(f"Preparing to write SRT to file: {output_srt_filename}")

    # Try to write subtitles to the SRT file.
    try:
        with open(output_srt_filename, 'w', encoding='utf-8') as f:
            f.write(srt.compose(subtitles))
        print("SRT file written successfully")
    except Exception as e:
        print(f"Error while writing SRT file: {e}")

    # Create a dictionary to hold additional JSON information
    extra_info = {
        'filename': videoFilePath,
        'date_processed': datetime.now().strftime("%Y-%m-%d-%H-%M"),
        'ocr_language': pytesseractLanguage,
        'analysis_frame_interval': args.frame_rate,
        'character_blacklist': args.blacklist
    }

    # Insert the extra_info dictionary at the beginning of the json_output list
    json_output.insert(0, extra_info)

    # Define the output filename for the JSON file.
    output_json_filename = videoFilePath.rsplit('.', 1)[0] + "_" + pytesseractLanguage + "_" + datetime.now().strftime(
        "%Y-%m-%d-%H-%M") + ".json"
    print(f"Preparing to write JSON to file: {output_json_filename}")

    # Try to write JSON data to the file.
    try:
        with open(output_json_filename, 'w') as json_file:
            json.dump(json_output, json_file, indent=4)
        print("JSON file written successfully")
    except Exception as e:
        print(f"Error while writing JSON file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract text from video using OCR and generate SRT file')
    parser.add_argument('-v', '--video', help='Path to the video file', required=True)
    parser.add_argument('-m', '--model', help='Path to the pre-trained EAST text detector model', required=True)
    parser.add_argument('-l', '--language', help='Language model for Pytesseract', default='eng')
    parser.add_argument('-f', '--frame_rate', help='Number of frames to skip for processing', type=int, default=10)
    parser.add_argument('-p', '--preview', help='Enable preview of the video with bounding boxes', action='store_true')
    parser.add_argument('-b', '--blacklist', help='blacklist characters to improve OCR result', default='@^¨#$«|{}_ı[]°<>»%=+´`§*')
    args = parser.parse_args()

    main(args)
