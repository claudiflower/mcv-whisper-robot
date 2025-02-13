# Automatic Speech Recognition Robot
The project is designed to run Whisper and Google Speech-to-Text API for the MCV listening experiments.

## Build
Create a Python virtual environment, activate it, `git clone` the project, and install the required dependencies.
```bash
python3 -m venv mcv-whisper-robot
source mcv-whisper-robot/bin/activate
cd mcv-whisper-robot
git clone git@github.com:irtlab/mcv-whisper-robot.git
pip3 install -r mcv-whisper-robot/requirements.txt
pip3 install git+https://github.com/openai/whisper.git 
pip install openai==0.28
pip install python-Levenshtein==0.12.0
```

To install Whisper, run the following command: (see instructions for more details: https://github.com/openai/whisper#setup)
```
pip3 install git+https://github.com/openai/whisper.git 
```

## Usage
Make sure you are are in the root directory of the source code. Now you can run the project by executing the following command
with the correct arguments.

**Arguments:**  
`--id`     - Experiment run ID (MongoDB `_id` field)  
`--engine` - Engine mode. There are two engine modes: `whisper` and `GoogleSTT` (Google Speech-to-Text)  
`--model`  - Whisper model. If the `--engine=whisper`, then the model must be set to `base`, `medium` or `large`

For example,
```bash
python speech_to_text.py --id=64822d9847d575f5c76aa2b9 --engine=whisper --model=medium
GOOGLE_STT_KEY=AIzaSyB-5VKtWsx7yCGCOxHRfpRDyZGjU8f4N80 
python speech_to_text.py --id=64822d9847d575f5c76aa2b9 --engine=GoogleSTT
```

## Running Output
The system creates an `output` directory and writes results in JSON files in the following file-name format:  
`<experiment run ID>_<engine name>_<model name if provided>.json`.  
For example, `64822d9847d575f5c76aa2b9_GoogleSTT.json`

Output JSON file format:
```JSON
{
  "id": "<experiment run ID>",
  "engine": "<engine>",
  "mode": "<if whisper engine is selected>",
  "blue_rate": "<Blue rate>",
  "combined_rate": "<combined rate>",
  "avg_normalized_lscore": "<average normalized lscore>"
}
```

