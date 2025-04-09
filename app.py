import gradio as gr
from langchain_huggingface import HuggingFacePipeline # <-- Use this from the dedicated package
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from datasets import load_dataset
from dotenv import load_dotenv

import warnings
import traceback

from config import REPO_ID, STORY_REPO_ID


warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub.repocard')
load_dotenv()

# --- LLM Configuration ---
llm = None
print("Attempting to initialize local HuggingFacePipeline (from langchain-huggingface)...")
print(f"This will download the model '{REPO_ID}' (~1GB) if not already cached.")
print("Please be patient, this might take a few minutes...")

try:
    llm = HuggingFacePipeline.from_model_id(
        model_id=REPO_ID,               
        task="text2text-generation",    
        pipeline_kwargs={               
            "max_new_tokens": 128,
            "temperature": 0.3,
            "do_sample": True,
        },
        device=0,                    
    )

    print(f"Successfully initialized HuggingFacePipeline for model: {REPO_ID}.")

except Exception as e:
    print("------------------------------------------------------")
    print(f"CRITICAL ERROR: Failed to initialize HuggingFacePipeline (from langchain-huggingface).")
    print(f"Specific Exception: {e}")
    print("Troubleshooting:")
    print("1. Ensure 'langchain-huggingface', 'transformers', 'torch' (or 'tensorflow'), 'sentencepiece' are installed.")
    print("2. Check if you have enough RAM/disk space for the model.")
    print("3. Verify the model ID '{REPO_ID}' and task 'text2text-generation' are correct.")
    print("4. If using GPU, check CUDA/driver setup.")
    print("------------------------------------------------------")
    llm = None


# --- Prompt Templates ---
genre_template = """
Analyze the following short story and determine its primary genre. Choose from common genres like: Science Fiction, Fantasy, Mystery, Thriller, Romance, Historical Fiction, Horror, Adventure, Drama, Comedy.

Story:
{story}

Primary Genre:
"""
genre_prompt = PromptTemplate(template=genre_template, input_variables=["story"])

summary_template = """
Provide a 2-3 sentences concise summary of the following story.

Story:
{story}

Summary:
"""
summary_prompt = PromptTemplate(template=summary_template, input_variables=["story"])

# --- Langchain Expression Language (LCEL) Chains --- 
if llm:
    print("LLM object exists, creating LCEL chains...")
    try:
        genre_chain = genre_prompt | llm | StrOutputParser()
        summary_chain = summary_prompt | llm | StrOutputParser()
        print("LCEL chains created successfully.")
    except Exception as e:
        print(f"Error creating LCEL chains: {e}")
        genre_chain = None
        summary_chain = None
else:
    print("LLM object is None, chains cannot be created.")
    genre_chain = None
    summary_chain = None


# --- Core Logic Function ---
def classify_and_summarize_story(story_text):
    if not llm or not genre_chain or not summary_chain:
        if not llm: return "Error: LLM initialization failed (local pipeline). Check logs.", "Initialization Error."
        else: return "Error: Chains not created. Check logs.", "Chain Error."

    if not story_text or not isinstance(story_text, str) or len(story_text.strip()) < 10:
        return "Please enter a valid story (at least 10 characters).", ""

    try:
        print(f"Invoking genre chain (local pipeline) for story length: {len(story_text)}")
        genre = genre_chain.invoke({"story": story_text})
        print(f"Raw genre output: '{genre}'")

        print(f"Invoking summary chain (local pipeline) for story length: {len(story_text)}")
        summary = summary_chain.invoke({"story": story_text})
        print(f"Raw summary output: '{summary}'")

        # --- Cleanup  ---
        if isinstance(genre, str):
            if genre.startswith("Primary Genre:"): 
                genre = genre.replace("Primary Genre:", "").strip()
        else: genre = "Invalid genre format received"

        if isinstance(summary, str):
            if summary.startswith("Summary:"): summary = summary.replace("Summary:", "").strip()
        else: summary = "Invalid summary format received"

        genre = genre if genre else "Could not determine genre"
        summary = summary if summary else "Could not generate summary"

        return genre.strip(), summary.strip()

    except Exception as e:
        print(f"Error during LLM processing/invocation (local pipeline): {e}")
        traceback.print_exc()
        error_message = str(e)
        return f"Processing Error (local). Check server logs.", f"Error: {error_message[:100]}..."

# --- Load Examples from Dataset ---
dataset_examples = []
try:
    print("Attempting to load dataset examples")
    story_dataset = load_dataset(STORY_REPO_ID, split="train[6:8]")
    dataset_examples = [[item['text']] for item in story_dataset]
    print(f"Loaded {len(dataset_examples)} examples.")
except Exception as e:
    print(f"INFO: Failed to load dataset 'allura-org/r_shortstories_24k' (Reason: {e}). Using fallback examples.")
    dataset_examples = [
        ["The old house stood on a hill overlooking the town..."],
        ["Commander Jax piloted the Star Hopper through the asteroid field..."],
        ["Lady Annelise adjusted her silk gown..."]
    ]
    print(f"Using {len(dataset_examples)} fallback examples.")


# --- Gradio Interface --- (Keep as is)
story_input = gr.Textbox(lines=15, placeholder="Paste or type your short story here...", label="Input Short Story")
genre_output = gr.Textbox(label="Detected Genre", interactive=False)
summary_output = gr.Textbox(label="Generated Summary", interactive=False)


iface = gr.Interface(
    fn=classify_and_summarize_story,
    inputs=story_input,
    outputs=[genre_output, summary_output],
    title="Automated Story Genre Classifier & Summarizer",
    description="""
    Enter a short story below.
    The tool uses the **google/flan-t5-base** model (running locally via HuggingFacePipeline) to:
    1.  **Classify** the story's primary genre.
    2.  **Generate** a concise summary.
    Examples below use fallback data. *Note: The first request might be slow as the local model loads.*
    """,
    examples=dataset_examples,
    flagging_mode='never',
    theme=gr.themes.Soft(),
)


# --- Launch the App --- (Keep as is, added transformers check)
if __name__ == "__main__":
    print("Launching Gradio Interface...")
    iface.launch(share=False)