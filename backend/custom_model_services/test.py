from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModel,
)
import traceback
import logging
import os

logger = logging.getLogger(__name__)

def validate_local_model_folder(folder_path):
    required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
    for file in required_files:
        if not os.path.exists(os.path.join(folder_path, file)):
            raise FileNotFoundError(f"Missing required file: {file} in {folder_path}")
    print("All required files are present.")


def load_model_and_tokenizer(
    model_name_or_path, 
    task="generation", 
    local_files_only=False
):
    try:
        logger.info(f"Loading model, tokenizer, and config for: {model_name_or_path}")
        
        # Load tokenizer with more robust settings
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            use_fast=True,  # Works with more models
            fallback_to_first=True  # Fallback to first available tokenizer
        )

        # Ensure the tokenizer has a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load configuration
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )

        # More flexible model loading
        if task == "generation":
            try:
                # First attempt: Causal Language Model
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    config=config,
                    local_files_only=local_files_only,
                )
            except Exception as causal_error:
                try:
                    # Fallback: Sequence-to-Sequence Language Model
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name_or_path,
                        config=config,
                        local_files_only=local_files_only,
                    )
                except Exception as seq2seq_error:
                    # Final fallback: Generic AutoModel
                    model = AutoModel.from_pretrained(
                        model_name_or_path,
                        config=config,
                        local_files_only=local_files_only,
                    )
        elif task == "classification":
            # Example: Load a generic AutoModel for classification tasks
            model = AutoModel.from_pretrained(
                model_name_or_path,
                config=config,
                local_files_only=local_files_only,
            )
        else:
            raise ValueError(f"Unsupported task: {task}")

        logger.info("Model, tokenizer, and config loaded successfully!")
        return model, tokenizer, config

    except Exception as e:
        logger.error(f"Error during loading: {traceback.format_exc()}")
        raise

def run_inference(model, tokenizer, input_text, task="generation", max_new_tokens=50):
    """
    Run inference on the model for a given input.

    Args:
        model: Hugging Face model object.
        tokenizer: Hugging Face tokenizer object.
        input_text (str): Input text for the model.
        task (str): Task type ("generation", "classification", etc.).
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        str: Generated or processed text.
    """
    try:
        # Encode the input text
        inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=True)

        if task == "generation":
            # Detect the appropriate generation method based on model type
            if hasattr(model, 'generate'):
                # Generate with more controlled parameters
                outputs = model.generate(
                    inputs['input_ids'], 
                    attention_mask=inputs.get('attention_mask'),
                    max_new_tokens=max_new_tokens,
                    do_sample=True,  # Enable sampling for more diverse output
                    temperature=0.7,  # Control randomness
                    top_p=0.9,        # Nucleus sampling
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    no_repeat_ngram_size=2  # Reduce repetition
                )
                
                # Decode, removing the input prompt
                full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_text = full_text[len(input_text):].strip()
                return generated_text
            else:
                # Fallback for models without generate method
                raise ValueError("Model does not support text generation")
        
        elif task == "classification":
            # Example: Process the outputs for classification tasks
            outputs = model(**inputs)
            return outputs.last_hidden_state  # Return last hidden state for classification
        else:
            raise ValueError(f"Unsupported task: {task}")
    except Exception as e:
        logger.error(f"Error during inference: {traceback.format_exc()}")
        raise


try:
    # prompt = "Summarize this: The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."
    prompt = "Tell me a story"

    # # Example: Locally downloaded model
    # print("------------------------------------------------------------------------------")
    # print("Testing gpt2 ")
    # local_path = "/Users/sumanthramesh/Documents/dev/cloud/LLM-Aggregation-Platform/backend/custom_model_services/custom_models/gpt2"
    # # config = AutoConfig.from_pretrained(local_path)
    # # print(config)
    # model, tokenizer, config = load_model_and_tokenizer(local_path, local_files_only=True)
    # result = run_inference(model, tokenizer, input_text=prompt)
    # print("Result: ", result)
    # print("------------------------------------------------------------------------------")


    # # Example: Locally downloaded model
    # print("------------------------------------------------------------------------------")
    # print("Testing gpt-neo-125m")
    # local_path = "/Users/sumanthramesh/Documents/dev/cloud/LLM-Aggregation-Platform/backend/custom_model_services/custom_models/gpt-neo copy"
    # model, tokenizer, config = load_model_and_tokenizer(local_path, local_files_only=True)
    # result = run_inference(model, tokenizer, input_text=prompt)
    # print("Result: ", result)
    # print("------------------------------------------------------------------------------")


    # print("------------------------------------------------------------------------------")
    # print("Testing lucadilipp bart small")
    # local_path = "lucadiliello/bart-small"
    # model, tokenizer, config = load_model_and_tokenizer(local_path, local_files_only=True)
    # result = run_inference(model, tokenizer, input_text=prompt)
    # print("Result: ", result)
    # print("------------------------------------------------------------------------------")

    # Example: Private model using Hugging Face token
    # private_model = "private-org/private-model"
    # model, tokenizer, config = load_model_and_tokenizer(private_model, use_auth_token=True)

    models_to_test = [
        "gpt2",
        "EleutherAI/gpt-neo-125M", 
        "google/flan-t5-small",
        # "microsoft/phi-2",
        # "lucadiliello/bart-small"
    ]

    for model_name in models_to_test:
        try:
            print(f"\nTesting {model_name}")
            model, tokenizer, config = load_model_and_tokenizer(model_name)
            result = run_inference(model, tokenizer, input_text=prompt)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")

except Exception as e:
    logger.error(f"Failed to load model: {e}")
