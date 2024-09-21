import gradio as gr
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
from groq import Groq

# Define the function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to run the user input code and display the plot
def run_code(code, groq_api_key):
    try:
        # Setup Groq API client with the provided API key
        llava_client = Groq(api_key=groq_api_key)
        llama_client = Groq(api_key=groq_api_key)

        fig, ax = plt.subplots()

        # Create a safe environment to execute code
        exec(code, {"plt": plt, "ax": ax})

        # Save the plot to a byte buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)

        # Open the saved image and resize it if necessary
        img = Image.open(buf)
        max_width, max_height = 600, 400  # Set maximum display size
        img.thumbnail((max_width, max_height))  # Resize to fit

        # Save the image to the disk
        img.save("plot.png")
        buf.seek(0)

        # Encode the image for Groq API
        base64_image = encode_image("plot.png")

        # Sending the plot image to LLava API to get the description
        llava_completion = llava_client.chat.completions.create(
            model="llava-v1.5-7b-4096-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Describe the plot values with image and the code provided to you. Code: {code}\n"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    },
                ],
            }],
            temperature=0.5,
            max_tokens=4096,
            top_p=1,
            stream=False,
            stop=None,
        )

        # Extract the LLava description from the API response
        llava_description = llava_completion.choices[0].message.content

        # Sending the plot image to Llama 3.1 API to get the description
        llama_completion = llama_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "What are the details of the plot provided by the user. Point out the important things in the dataset. What is the purpose of this dataset and how to interpret it. Analyze it."
                },
                {
                    "role": "user",
                    "content": code
                }
            ],
            temperature=0,
            max_tokens=4096,
            top_p=1,
            stream=True,
            stop=None,
        )

        # Extract the Llama 3.1 description from the API response
        llama_description = ""
        for chunk in llama_completion:
            llama_description += chunk.choices[0].delta.content or ""

        return img, llava_description, llama_description

    except Exception as e:
        return None, f"Error: {str(e)}", None
    finally:
        plt.close(fig)

# Define the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""## Plot and Describe - Inference Powered by [Groq](https://groq.com/)

    **⚠️ Disclaimer:** Generative models may hallucinate or produce incorrect outputs. This tool is built for demonstration purposes only and should not be relied upon for critical analysis or decision-making.
    """, elem_id="disclaimer")

    with gr.Row():
        api_key_input = gr.Textbox(
            label="Groq API Key", type="password", placeholder="Enter your Groq API key here"
        )

    with gr.Row():
        code_input = gr.Code(
            language="python", lines=20, label="Input Code"
        )
        output_image = gr.Image(type="pil", label="Chart will be displayed here")

    submit_btn = gr.Button("Submit")

    with gr.Row():
        output_llava_text = gr.Textbox(label="Description from LLaVA", interactive=False)
        output_llama_text = gr.Textbox(label="Description from Llama 3.1", interactive=False)

    submit_btn.click(
        fn=run_code,
        inputs=[code_input, api_key_input],
        outputs=[output_image, output_llava_text, output_llama_text]
    )

# Launch the interface
demo.launch()
