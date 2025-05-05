import torch
#import torch_maia
#import maia_athena
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st
import time

status_placeholder = st.empty()
status_placeholder.info('Model is loading...')

# Load the Maia firmware and increase the global HBM domain size 
# so that the full model weights can be copied to the device

#torch_maia.load_firmware(0)
#maia_athena.get_nepal_device(0).set_global_hbm_limit(int(40e9))
#device="maia"

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", attn_implementation="eager", revision="26bca36bde8333b5d7f72e9ed20ccda6a618af24")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", revision="26bca36bde8333b5d7f72e9ed20ccda6a618af24")
    return model, tokenizer


def generate_response(prompt):
    # Tokenize the input prompt
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    # Move the model to the device
    model.to(torch.bfloat16)
    model.to(device)
    # Generate the response
    generated_ids = model.generate(**model_inputs, max_new_tokens=10, do_sample=True)
    # Decode the generated response
    result = tokenizer.batch_decode(generated_ids)[0]
    return result


# Download and cache the model weights on the node. It requires the HF_TOKEN environment variable to be set.
model, tokenizer = load_model()
status_placeholder.success('Model loaded. Ready.')

st.title("Mistral Model Playground")
prompt = st.text_area("Enter your prompt:")
generate_button = st.button("Generate")
st.markdown("**Model Output:**")
output = st.empty()


if generate_button:
    if prompt.strip():
        with st.spinner('Model is generating response...'):
            start_time = time.time()
            # Generate the response
            result = generate_response(prompt)
            # Calculate elapsed time
            elapsed = time.time() - start_time

            # Display the result
            st.markdown("**Result:**")
            st.write(result)
            st.info(f"Time taken to generate response: {elapsed:.2f} seconds")
            print(result)
        status_placeholder.success('Model finished generating. Ready.')
    else:
        st.warning("Please enter a prompt.")
