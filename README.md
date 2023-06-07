This repository contains a Python script that performs Optical Character Recognition (OCR) on video files and generates SRT subtitle files. The script leverages the [EAST text detector](https://arxiv.org/abs/1704.03155v2) model for text detection and the Pytesseract library for OCR.

### Current state/things to improve
- Currently the the OCR is written to a SRT per processed frame. The same text will be added as multiple subtitle objects for every processed frame. A better behaviour would be to use a single subtitle element for the duration of similar text. 
  - I've added an alternative version of the script called video-ocr2srt_fuzzy.py which uses simple string matching (through fuzzywuzzy) to combine similar subtitle elements into continuous longer elements. It works quite well, but does not utlize the multiple ocr readings to produce better results. Currently it only keeps the first ocr string and extends it for the duration of matched following strings.
- Reduce amount of false positives! (character white-/blacklists? confidence tweaks?)

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
- `-p` or `--preview`: This argument is optional. If included, it enables a preview of the video that highligts when text is detected.

To process a video file named `video.mp4` with the EAST model `east_model.pb`, you would use the following command:

```sh
python video-ocr2srt.py -v video.mp4 -m east_model.pb
```

The script will process the video, performing OCR on the specified frames, and will output an SRT subtitle file with the same name as the input video file, in the format `<video_filename>_<language>_<timestamp>.srt`.
