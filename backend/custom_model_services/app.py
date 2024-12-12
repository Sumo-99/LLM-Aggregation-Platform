from flask import Flask, request, jsonify
import logging
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from huggingface_hub import login
import os
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL = None  # Hugging Face model object
TOKENIZER = None  # Hugging Face tokenizer object
CONFIG = None  # Hugging Face configuration object
ARCHITECTURE = None  # Store the architecture type provided during initialization


def download_model_from_s3(bucket_name, object_name):

    model_path = object_name

    s3_url = f"https://{bucket_name}.s3.amazonaws.com/{object_name}"
    logger.info(f"Constructed S3 URL: {s3_url}")

    if not os.path.exists(model_path):
        logger.info(f"Downloading model from {s3_url}...")
        os.system(f"curl -o {model_path} {s3_url}")
        logger.info("Model downloaded successfully!")
    else:
        logger.info(f"Model already exists locally: {model_path}")

    return model_path


def load_model_and_tokenizer(architecture, model_path):

    global MODEL, TOKENIZER, CONFIG
    logger.info(f"Loading model and tokenizer for architecture: {architecture}...")

    try:
        if architecture == "bart":
            model_name = "lucadiliello/bart-small"
            TOKENIZER = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
            CONFIG = AutoConfig.from_pretrained(model_name, use_auth_token=True)
            MODEL = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=True)
        elif architecture == "llama":
            logger.info(f"Using locally downloaded model path: {model_path}")
            TOKENIZER = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            CONFIG = AutoConfig.from_pretrained(model_path, local_files_only=True)
            MODEL = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        logger.info(f"Model, tokenizer, and config loaded successfully for {architecture.capitalize()}!")
    except Exception as e:
        logger.error(f"Error during loading: {traceback.format_exc()}")
        raise


@app.route('/init', methods=['POST'])
def init_model():

    global ARCHITECTURE
    login(token='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')
    data = request.get_json()
    if not data or 'bucket_name' not in data or 'object_name' not in data or 'architecture' not in data:
        logger.error("Missing 'bucket_name', 'object_name', or 'architecture' in an data")
        return jsonify({"error": "Missing 'bucket_name', 'object_name', or 'architecture' in request data"}), 400

    bucket_name = data['bucket_name']
    object_name = data['object_name']
    ARCHITECTURE = data['architecture']
    logger.info(f"Initializing model with architecture: {ARCHITECTURE}")

    try:
        model_path = download_model_from_s3(bucket_name, object_name)
        load_model_and_tokenizer(ARCHITECTURE, model_path)
        print("model path ---> ", model_path)
        print("folder ---> ", os.path.dirname(model_path))
        return jsonify({"message": f"{ARCHITECTURE.capitalize()} model initialized successfully!"})
    except Exception as e:
        logger.error(f"Error during model initialization: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/process', methods=['POST'])
def process_request():

    data = request.get_json()
    if not data or 'input_text' not in data:
        logger.error("Missing 'input_text' in request data")
        return jsonify({"error": "Missing 'input_text' in request data"}), 400

    input_text = data['input_text']
    logger.info(f"Processing input text: {input_text}")

    try:
        if ARCHITECTURE in ["bart", "llama"]:
            # Tokenize input and generate output
            inputs = TOKENIZER.encode(input_text, return_tensors="pt")
            outputs = MODEL.generate(inputs, max_length=50, num_beams=5, early_stopping=True)
            generated_text = TOKENIZER.decode(outputs[0], skip_special_tokens=True)

            logger.info(f"Generated response: {generated_text}")
            return jsonify({"response": generated_text})
        else:
            logger.error(f"Unsupported architecture: {ARCHITECTURE}")
            return jsonify({"error": f"Unsupported architecture: {ARCHITECTURE}"}), 400
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/')
def hello_world():
    return "Dynamic Model Loading API with AutoTokenizer and AutoConfig is running!"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)