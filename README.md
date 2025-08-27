# Databricks HuggingFace Model Deployment

A complete workflow for pulling models from Hugging Face, saving them to Databricks Unity Catalog Volumes, and deploying them on Databricks Model Serving for use as REST APIs.

# Disclaimer
This is not an official Databricks asset. There are no guarantees and code is subject to change. Use as reference at your own risk. 

## Overview

This project provides a streamlined three-step process to deploy Hugging Face models on Databricks:

1. **Create Secrets** - Set up HuggingFace authentication tokens
2. **Download Model** - Pull models from Hugging Face and cache them in Databricks Volumes
3. **Deploy Model** - Register and deploy the model to Databricks Model Serving with REST API access

The default configuration is set up for medical AI models (specifically `google/medgemma-27b-text-it`), but can be adapted for any Hugging Face model.

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- GPU-enabled compute cluster for model inference
- HuggingFace account and access token
- Appropriate permissions for:
  - Creating Unity Catalog volumes
  - Managing secrets
  - Creating model serving endpoints

## Project Structure

```
dbx_hf_model_deploy/
├── configs/
│   └── config.yaml                    # Central configuration file
├── notebooks/
│   ├── 00_create_secrets.ipynb        # Step 1: Setup HF authentication
│   ├── 01_download_model.ipynb        # Step 2: Download and cache model
│   ├── 02_deploy_model_dbx.ipynb      # Step 3: Deploy to model serving
│   └── requirements.txt               # Python dependencies
└── README.md
```

## Quick Start

### 1. Configuration Setup

Edit `configs/config.yaml` with your specific values:

```yaml
catalog_name: "your_catalog"           # Unity Catalog name
schema_name: "your_schema"             # Schema for model storage
volume_name: "your_volume"             # Volume for model files
volume_folder: "model_cache"           # Folder within volume
model_name: "google/medgemma-27b-text-it"  # HuggingFace model ID
revision: "6b08c481126ff65a9b8fa5ab4d691b152b8edb5d"  # Model revision/commit
secret_scope: "your_scope"             # Databricks secret scope
secret_key: "hf_token"                 # Secret key name
served_model_name: "your_served_model" # Name for serving endpoint
```

### 2. Run Notebooks in Order

#### Step 1: Create Secrets (`00_create_secrets.ipynb`)
- Creates a Databricks secret scope
- Stores your HuggingFace token securely
- **Update the `secret_value` variable with your actual HuggingFace token**

#### Step 2: Download Model (`01_download_model.ipynb`)
- Authenticates with HuggingFace using stored secrets
- Downloads the specified model and tokenizer
- Supports quantization options (4-bit, 8-bit) for memory efficiency
- Caches model files in Databricks Unity Catalog Volumes

#### Step 3: Deploy Model (`02_deploy_model_dbx.ipynb`)
- Loads cached model from Unity Catalog Volumes
- Creates MLflow PyFunc wrapper for serving
- Registers model in Unity Catalog Model Registry
- Deploys to Databricks Model Serving with GPU inference
- Creates REST API endpoint for model inference

## Features

### Model Optimization
- **Quantization Support**: Optional 4-bit or 8-bit quantization for memory efficiency
- **GPU Acceleration**: Optimized for CUDA devices with bfloat16 precision
- **Caching Strategy**: Efficient model caching using Databricks Volumes

### MLflow Integration
- **Model Versioning**: Automatic model versioning with MLflow
- **Model Registry**: Integration with Unity Catalog Model Registry
- **Model Aliases**: Support for Champion/Challenger model aliases

### Production Ready
- **REST API**: Automatic REST endpoint creation
- **Auto Scaling**: Scale-to-zero enabled for cost optimization
- **Monitoring**: Optional inference logging and monitoring
- **GPU Serving**: Large GPU workload support for inference

## API Usage

Once deployed, the model serves as a REST API that accepts JSON requests:

```json
{
  "dataframe_records": [
    {
      "system_prompt": "You are a helpful medical assistant.",
      "user_prompt": "What are the symptoms of hypertension?"
    }
  ],
  "params": {
    "max_tokens": 1024
  }
}
```

## Dependencies

The project requires the following Python packages (see `notebooks/requirements.txt`):

- `transformers==4.55.4` - HuggingFace transformers library
- `accelerate==1.10.1` - Model acceleration utilities
- `huggingface_hub==0.34.4` - HuggingFace Hub client
- `torch` - PyTorch for model inference
- `mlflow` - Model tracking and serving
- `databricks-sdk` - Databricks workspace integration

## Customization

### For Different Models
1. Update `model_name` and `revision` in `config.yaml`
2. Modify the prompt template in `02_deploy_model_dbx.ipynb` (HFModelPyfunc class)
3. Adjust quantization settings based on model size and memory requirements

### For Non-Medical Models
- Update system prompts in the PyFunc model class
- Modify input/output schema as needed
- Adjust inference parameters (temperature, top_p, etc.)

## Troubleshooting

- **Out of Memory**: Enable quantization in notebook 01 or use larger compute instances
- **Authentication Issues**: Verify HuggingFace token has proper model access permissions
- **Volume Access**: Ensure proper Unity Catalog permissions for volume creation and access
- **Serving Deployment**: Check GPU availability and cluster permissions

## License

See `LICENSE.txt` for license information.
