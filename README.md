# PDF Manual Processing App

This project provides tools to process PDF manuals using LLMs, available as both a CLI tool and a Streamlit web application.

## Features

- **PDF to Text/Markdown**: Extracts text and visual descriptions from PDFs.
- **Streamlit Web Interface**: User-friendly interface for uploading and processing PDFs.
- **CLI Support**: Batch processing via command line.
- **Dockerized**: Easy deployment using Docker.

## Getting Started

### Prerequisites

- Docker
- Python 3.11+ (if running locally without Docker)
- OpenAI API Key

### Running with Docker (Recommended)

1.  **Build the image:**
    ```bash
    make -f Makefile.dev build
    ```

2.  **Run the application:**
    ```bash
    make -f Makefile.dev run
    ```
    The application will be available at `http://localhost:8501`.

### Running Locally

1.  Installed dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Project Structure

- `app.py`: Streamlit web application entry point.
- `process_pdf_cli.py`: Command-line interface for PDF processing.
- `Dockerfile` / `Dockerfile.dev`: Docker configuration.
- `requirements.txt`: Python dependencies.
