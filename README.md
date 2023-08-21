This repository contains a Python script that performs Optical Character Recognition (OCR) on video files and generates an SRT subtitle file as well as a more detailed JSON file. 

Current functionality has been focused on getting good text readings from intertitles in silent films, rather than text in frames in general. For more general text extraction, the model will have to be tweaked to accept skewewd and angled text.

The script leverages the [EAST text detector](https://arxiv.org/abs/1704.03155v2) model for text detection and the Pytesseract library for OCR.

### Current state
- Currently two files are output: 
  - A JSON file that includes some basic information about the file and parameters used, as well as OCR strings marked with frame number, timecode (in ms) and confidence scores for text detection and the actual OCR process. 
  - A functional SRT file for the processed video file.
    - Currently the OCR strings are written to the SRT per processed frame. The same text will be added as multiple subtitle objects for every processed frame. 

### Things to improve
- SRT creation using single subtitle elements for the duration of similar text. I've added an alternative version of the script called video-ocr2srt_fuzzy.py which uses simple string matching (through fuzzywuzzy) to combine similar subtitle elements into continuous longer elements. It works ok, but does not utlize the multiple ocr readings to produce better results. Currently it only keeps the first ocr string and extends it for the duration of matched following strings.
- Reduce amount of false positives! 
- Better perfomance on odd fonts/handwritten text. (evaluate other OCR libraries/fine-tune a model on silent film fonts)

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
- `-b` or `--blacklist`: This argument is optional. If included it lets you specify characters to blacklist in OCR. By default these characters are blacklisted: "@^¨#$«»|{} ı[]°<>»%=+´`§*_"

To process a video file named `video.mp4` with the text detection model `east_model.pb`, you would use the following command:

```sh
python video-ocr2srt.py -v video.mp4 -m east_model.pb
```

The script will process the video, performing OCR on the specified frames, and will output an SRT subtitle file with the same name as the input video file, in the format `<video_filename>_<language>_<timestamp>.srt`.
