from huggingface_hub import HfApi

api = HfApi()

# Upload pipe.pkl
api.upload_file(
    path_or_fileobj="pipe.pkl",
    path_in_repo="models/pipe.pkl",
    repo_id="ByteBard101/TransitIQ", 
    repo_type="model"
)

# Upload column_names.pkl
api.upload_file(
    path_or_fileobj="column_names.pkl",
    path_in_repo="models/column_names.pkl",
    repo_id="ByteBard101/TransitIQ", 
    repo_type="model"
)