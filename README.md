# PDF Processing System

A machine learning-based PDF document processing system that extracts layouts, classifies headings, and generates structured outputs using a trained local model.

## Overview

This system processes PDF documents through three phases:

1. **Phase 1**: Generate training data from sample PDFs
2. **Phase 2**: Train a local machine learning model
3. **Phase 3**: Process new PDFs using the trained model (current implementation)

## File Structure

```
project/
├── main.py                     # Main entry point for Phase 3 processing
├── utils/
│   ├── local_model.py         # Local ML model implementation
│   ├── layout_utils.py        # PDF layout extraction utilities
│   └── postprocess.py         # Post-processing and output generation
├── input/                     # Directory for input PDF files
├── output/                    # Directory for processed results (JSON files)
├── layout_data/              # Temporary directory for layout extraction data
├── models/                   # Directory for trained ML models (created during Phase 2)
├── Dockerfile                # Docker configuration
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Setup and Installation

### Local Installation

1. Clone the repository and navigate to the project directory
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Docker Setup

1. Build the Docker image:

   ```bash
   docker build -t pdf-processor .
   ```

2. Create the necessary directories on your host machine:
   ```bash
   mkdir -p ./input ./output ./models
   ```

## Usage

### Prerequisites

Before running Phase 3, ensure you have:

- A trained model from Phase 2 (should be saved in the `models/` directory)
- PDF files to process in the `input/` directory

### Cleaning Input Directory

To remove all previous PDF files before running a new test case:

```bash
# Linux/Mac
rm -f ./input/*.pdf

# Windows
del /q .\input\*.pdf

# Using Docker (if container is running)
docker exec <container-name> rm -f /app/input/*.pdf
```

### Running the Application

#### Local Execution

```bash
python main.py
```

#### Docker Execution

**Option 1: Using volume mounts (recommended)**

```bash
docker run -it --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/models:/app/models \
  pdf-processor
```

**Option 2: Copy files to container**

```bash
# Start container
docker run -it --name pdf-processor-container pdf-processor

# Copy input files
docker cp ./input/your-file.pdf pdf-processor-container:/app/input/

# Copy trained model (if not using volumes)
docker cp ./models/ pdf-processor-container:/app/

# Execute processing
docker exec pdf-processor-container python main.py

# Copy results back
docker cp pdf-processor-container:/app/output/ ./
```

**Option 3: Docker Compose (create docker-compose.yml)**

```yaml
version: "3.8"
services:
  pdf-processor:
    build: .
    volumes:
      - ./input:/app/input
      - ./output:/app/output
      - ./models:/app/models
    command: python main.py
```

Run with:

```bash
docker-compose up
```

## Input and Output

### Input

- **Location**: `./input/` directory
- **Format**: PDF files (\*.pdf)
- **Behavior**: All PDF files in the input directory will be processed

### Output

- **Location**: `./output/` directory
- **Format**: JSON files with the same name as input PDFs
- **Content**: Structured data with heading classifications and layout information

## Workflow

1. **Place PDF files** in the `./input/` directory
2. **Clean previous files** (optional) using the commands above
3. **Run the application** using one of the execution methods
4. **Check results** in the `./output/` directory

### Example Workflow

```bash
# Clean previous test files
rm -f ./input/*.pdf

# Add new PDF files to input directory
cp /path/to/your/document.pdf ./input/

# Run processing with Docker
docker run -it --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/models:/app/models \
  round-1a

# Check results
ls -la ./output/
```

## System Requirements

- Python 3.8+
- Docker (for containerized execution)
- Sufficient disk space for PDF processing and model storage
- Trained ML model from Phase 2

## Troubleshooting

### Common Issues

1. **"Failed to load the local model"**

   - Ensure Phase 2 has been completed and model files exist in `./models/`
   - Check file permissions on the models directory

2. **"No new PDFs found"**

   - Verify PDF files are in the correct input directory
   - Check file permissions and naming (must have .pdf extension)

3. **Docker volume mount issues**
   - Ensure directories exist on host machine before mounting
   - Check Docker permissions for volume access

### Logs and Debugging

The application provides console output showing:

- Number of PDFs found
- Processing progress for each file
- Success/failure messages
- Output file locations

## Notes

- The system processes all PDF files in the input directory in a single run
- Each PDF generates a corresponding JSON file with the same basename
- Layout data is temporarily stored in `./layout_data/` during processing
- The trained model must be available before running Phase 3

## Future Enhancements

- Support for batch processing configuration
- Integration of Phases 1 and 2 for complete pipeline execution
- Enhanced error handling and recovery mechanisms
- Support for additional output formats
