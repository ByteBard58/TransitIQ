from huggingface_hub import hf_hub_download as download_hf
import pickle

def download() -> None:
# Download the serialized pipeline 
  download_hf(
    repo_id="ByteBard101/TransitIQ",
    filename="models/pipe.pkl",
    repo_type="model",
    local_dir="."
  )

  # Download the serialized column names file
  download_hf(
    repo_id="ByteBard101/TransitIQ",
    filename="models/column_names.pkl",
    repo_type="model",
    local_dir="."
  )

if __name__ == "__main__":
  download()