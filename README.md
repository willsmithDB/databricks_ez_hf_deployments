# Databricks HuggingFace Model Deployment

Complete workflows for deploying different types of Hugging Face models on Databricks, including chat models and OCR models with specialized serving approaches.

# Disclaimer
**This is not an official Databricks asset. There are no guarantees and code is subject to change. Use as reference at your own risk.**

## Special Thanks
Thank you to srijit.nai@databricks.com and eli.swanson@databricks.com for all of your help with work preceding this repo! 

## Overview

This project provides two distinct deployment workflows for Hugging Face models on Databricks:

### 1. Chat Model Workflow (Standard Model Serving)
A streamlined three-step process to deploy chat/text generation models:
1. **Create Secrets** - Set up HuggingFace authentication tokens
2. **Download Model** - Pull models from Hugging Face and cache them in Databricks Volumes  
3. **Deploy Model** - Register and deploy to Databricks Model Serving with REST API access

### 2. vLLM OCR Model Workflow 
A specialized workflow for OCR (Optical Character Recognition) models using vLLM for high-performance inference:
1. **Download OCR Model** - Pull OCR models (like AllenAI's OLMoCR-7B) and cache them
2. **Test Inference** - Validate OCR functionality with document/image processing
3. **Deploy with vLLM** - Deploy using vLLM for optimized OCR model serving

The chat model workflow defaults to medical AI models (e.g., `google/medgemma-27b-text-it`) while the OCR workflow is configured for document understanding models (e.g., `allenai/olmOCR-7B-0825`).

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
│   ├── config.yaml                    # Chat model configuration
│   ├── medgemma_config.yaml           # MedGemma specific config
│   ├── olmocr_config.yaml             # OLMoCR OCR model config
│   └── sample_config.yaml             # Template configuration
├── setup/
│   ├── 00_create_secrets.ipynb        # Setup HF authentication (shared)
│   └── 01_unity_catalog_assets.ipynb  # Unity Catalog setup
├── chat_model_notebooks/              # Standard chat model workflow
│   ├── 01_download_model.ipynb        # Download and cache chat model
│   ├── 02_deploy_model_dbx.ipynb      # Deploy to standard model serving
│   └── requirements.txt               # Chat model dependencies
├── vllm_ocr_model_notebooks/          # vLLM OCR model workflow
│   ├── 01_download_model.ipynb        # Download and cache OCR model
│   ├── 02_test_inference.ipynb        # Test OCR inference locally
│   ├── 03_deploy_vllm.ipynb           # Deploy using vLLM serving
│   ├── requirements.txt               # OCR-specific dependencies  
│   └── conda.yaml                     # Conda environment spec
└── README.md
```

## Quick Start

Choose your deployment workflow based on your use case:

## Workflow 1: Chat Models (Standard Model Serving)

### 1. Configuration Setup

Edit `configs/config.yaml` (or `configs/medgemma_config.yaml`) with your specific values:

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

### 2. Run Chat Model Notebooks in Order

#### Step 1: Create Secrets (`setup/00_create_secrets.ipynb`)
- Creates a Databricks secret scope
- Stores your HuggingFace token securely
- **Update the `secret_value` variable with your actual HuggingFace token**

#### Step 2: Download Model (`chat_model_notebooks/01_download_model.ipynb`)
- Authenticates with HuggingFace using stored secrets
- Downloads the specified model and tokenizer
- Supports quantization options (4-bit, 8-bit) for memory efficiency
- Caches model files in Databricks Unity Catalog Volumes

#### Step 3: Deploy Model (`chat_model_notebooks/02_deploy_model_dbx.ipynb`)
- Loads cached model from Unity Catalog Volumes
- Creates MLflow PyFunc wrapper for serving
- Registers model in Unity Catalog Model Registry
- Deploys to Databricks Model Serving with GPU inference
- Creates REST API endpoint for model inference

## Workflow 2: OCR Models with vLLM

### 1. Configuration Setup

Edit `configs/olmocr_config.yaml` with your specific values:

```yaml
catalog_name: "your_catalog"           # Unity Catalog name
schema_name: "your_schema"             # Schema for model storage
volume_name: "your_volume"             # Volume for model files
volume_folder: "model_cache"           # Folder within volume
model_name: "allenai/olmOCR-7B-0825"   # OLMoCR model for OCR tasks
revision: "e14b3cfbf2f0b85ed95cac9c86017f7abc3e7194"  # Model revision/commit
secret_scope: "your_scope"             # Databricks secret scope  
secret_key: "hf_token"                 # Secret key name
served_model_name: "your_served_model" # Name for serving endpoint
```

### 2. Run vLLM OCR Notebooks in Order

#### Step 1: Download OCR Model (`vllm_ocr_model_notebooks/01_download_model.ipynb`)
- Installs required system dependencies (poppler-utils for PDF processing)
- Downloads AllenAI's OLMoCR-7B model optimized for OCR tasks
- Caches model files in Databricks Unity Catalog Volumes
- Includes image/document processing dependencies

#### Step 2: Test Inference (`vllm_ocr_model_notebooks/02_test_inference.ipynb`)  
- Validates OCR model functionality with sample documents/images
- Tests document understanding and text extraction capabilities
- Verifies model performance before deployment

#### Step 3: Deploy with vLLM (`vllm_ocr_model_notebooks/03_deploy_vllm.ipynb`)
- Deploys OCR model using vLLM for high-performance inference
- Optimized for document processing and OCR workloads
- Creates specialized endpoints for document understanding tasks

## Features

### Dual Deployment Workflows
- **Chat Models**: Standard Databricks Model Serving for conversational AI models
- **OCR Models**: Specialized vLLM deployment for document understanding and OCR tasks
- **Flexible Configuration**: Separate config files for different model types and use cases

### Model Optimization
- **Quantization Support**: Optional 4-bit or 8-bit quantization for memory efficiency (chat models)
- **GPU Acceleration**: Optimized for CUDA devices with bfloat16 precision
- **Caching Strategy**: Efficient model caching using Databricks Volumes
- **vLLM Integration**: High-performance serving for OCR models with optimized inference

### OCR Capabilities
- **Document Processing**: PDF and image text extraction using OLMoCR-7B
- **Vision-Language Understanding**: Advanced document comprehension and analysis
- **Multi-format Support**: PDF, image, and document processing with poppler-utils
- **Inference Testing**: Built-in testing notebooks for validation before deployment

### MLflow Integration (Chat Models)
- **Model Versioning**: Automatic model versioning with MLflow
- **Model Registry**: Integration with Unity Catalog Model Registry
- **Model Aliases**: Support for Champion/Challenger model aliases

### Production Ready
- **REST API**: Automatic REST endpoint creation for both workflows
- **Auto Scaling**: Scale-to-zero enabled for cost optimization
- **Monitoring**: Optional inference logging and monitoring
- **GPU Serving**: Large GPU workload support for inference
- **Specialized Serving**: vLLM for OCR, standard serving for chat models

## API Usage

### Chat Model API
Once deployed, chat models serve as REST APIs that accept JSON requests:

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

### OCR Model API
OCR models deployed with vLLM accept document/image inputs for text extraction and understanding:

```json
{
  "inputs": {
    "image_path": "/path/to/document.pdf",
    "prompt": "Extract and summarize the key information from this document"
  },
  "parameters": {
    "max_tokens": 2048,
    "temperature": 0.1
  }
}
```

## Dependencies

### Chat Model Dependencies
The chat model workflow requires the following packages (see `chat_model_notebooks/requirements.txt`):

- `transformers==4.55.4` - HuggingFace transformers library
- `accelerate==1.10.1` - Model acceleration utilities
- `huggingface_hub==0.34.4` - HuggingFace Hub client
- `torch` - PyTorch for model inference
- `mlflow` - Model tracking and serving
- `databricks-sdk` - Databricks workspace integration

### vLLM OCR Model Dependencies
The OCR model workflow requires additional packages (see `vllm_ocr_model_notebooks/requirements.txt`):

- `transformers==4.55.4` - HuggingFace transformers library
- `accelerate==1.10.1` - Model acceleration utilities
- `huggingface_hub==0.34.4` - HuggingFace Hub client
- `Pillow==11.3.0` - Image processing library
- `flask==3.1.2` - Web framework for model serving
- `bitsandbytes==0.47.0` - Quantization library
- `pdf2image==1.17.0` - PDF to image conversion
- **System Dependencies**: `poppler-utils` (installed via apt) for PDF processing

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

### General Issues
- **Out of Memory**: Enable quantization in download notebooks or use larger compute instances
- **Authentication Issues**: Verify HuggingFace token has proper model access permissions
- **Volume Access**: Ensure proper Unity Catalog permissions for volume creation and access

### Chat Model Issues
- **Serving Deployment**: Check GPU availability and cluster permissions for standard model serving
- **MLflow Errors**: Verify Unity Catalog Model Registry permissions

### OCR Model Issues
- **PDF Processing**: Ensure poppler-utils is properly installed (`apt-get install -y poppler-utils`)
- **Image Dependencies**: Verify Pillow and pdf2image are correctly installed
- **vLLM Deployment**: Check vLLM-specific GPU requirements and cluster configurations
- **Document Format**: Ensure input documents are in supported formats (PDF, PNG, JPG, etc.)

## License

See `LICENSE.txt` for license information.
