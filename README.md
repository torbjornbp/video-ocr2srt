This repository contains a Python script that performs Optical Character Recognition (OCR) on video files and generates SRT subtitle files. The script leverages the [EAST text detector](https://arxiv.org/abs/1704.03155v2) model for text detection and the Pytesseract library for OCR.

### Prerequisites

- Python 3.8 or higher
- Libraries and packages: argparse, srt, pytesseract, cv2, imutils, numpy, tqdm

Additionally, you will need to have the EAST text detector model file. You can download it from this [link](https://github.com/argman/EAST) or find it in the model directory of this repository.

### Usage

```sh
python video-ocr2srt.py -v <path_to_video> -m <path_to_model> [-l <language>] [-f <frame_rate>] [-p]
```

Where:

- `<path_to_video>`: This argument is required and it should be the path to the video file you want to process.
- `<path_to_model>`: This argument is required and it should be the path to the EAST text detector pre-trained model file.
- `<language>`: This argument is optional and it should be the language model for Pytesseract. The default is `eng` (English). You can provide other language codes supported by Tesseract.
- `<frame_rate>`: This argument is optional and it specifies the number of frames to skip for processing. By default, this value is `10`, which seems to give an ok compromise between detected text and and processing speed.
- `-p` or `--preview`: This argument is optional. If included, it enables the preview of the video whuch shows when text is detected.

To process a video file named `video.mp4` with the EAST model `east_model.pb`, you would use the following command:

```sh
python video-ocr2srt.py -v video.mp4 -m east_model.pb
```

The script will process the video, performing OCR on the specified frames, and will output an SRT subtitle file with the same name as the input video file, in the format `<video_filename>_<language>_<timestamp>.srt`.

### Things to improve
- Currently the the OCR is written to a SRT per processed frame. so the same text will be added as multiple subitle objects for every frame rather than the span of the subtitle. Duplicate subtitles can probably be merged somehow
- Reduce amount of false positives
