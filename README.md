

# Real-Time Translation

## Overview
This project aims to provide real-time translation services using a combination of speech recognition, machine translation, and text-to-speech technologies. It integrates several models and tools to achieve seamless communication across Chinese and English languages

## Repository Structure
- `melo/`
- `openvoice/`
- `seamless_communication/`
- `finetune_seamless_m4t_medium.ipynb`
- `seamless_translate.py`
- `tri-model_translation.py`

### Core Functionality

#### melo/
 
[Orignally from MyshelAI's MELOTTS project. Customized for this task.](https://github.com/myshell-ai/MeloTTS) 

Contains utilities and APIs for text normalization and text-to-speech (TTS) services.

#### openvoice/
 
[Orignally from MyshelAI's OPENVOICE project. Customized for this task.](https://github.com/myshell-ai/Openvoice)
 
Customization Features:
 
1. Removed Watermark Generation to provide a more faster interference time 
2. Removed Japanese, Spanish, French, and Korea to improve the initialization time (since this project is only Chinese to English)

Includes components for voice processing and manipulation, such as speaker extraction and tone color conversion.

#### seamless_communication/
 
[Orignally from facebook's SEAMLESS COMMUNICATION project. Customized for this task.](https://github.com/facebookresearch/seamless_communication)
 
Customization Features:
1. Customized Training Data
2. Customized Training Data, Val Data dateset class

Focuses on integrating different modules for seamless communication, including managing audio input/output and coordinating the translation pipeline.

#### finetune_seamless_m4t_medium.ipynb
A Jupyter Notebook for fine-tuning the Seamless M4T model, providing an environment for customizing the model to improve performance on specific datasets.

#### seamless_translate.py
Main script to perform translation tasks. It initializes and manages the translation pipeline, which includes speech recognition, translation, and text-to-speech conversion.

#### tri-model_translation.py
Script that integrates multiple models for enhanced translation accuracy. It includes functionalities for real-time speech recognition, translation, and TTS using various pre-trained models.

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- Transformers library by Hugging Face
- Additional dependencies listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/ivanhe123/real_time_translation.git
   cd real_time_translation
   ```

2. Install the dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Running the Project
1. **Fine-tuning the Model:**
   Open `finetune_seamless_m4t_medium.ipynb` in Jupyter Notebook and follow the instructions to fine-tune the model on your dataset.

2. **Real-Time Translation:**
   Run `seamless_translate.py` to start the translation pipeline:
   ```sh
   python seamless_translate.py
   ```

3. **Multi-Model Translation:**
   Run `tri-model_translation.py` to use the integrated multi-model approach:
   ```sh
   python tri-model_translation.py
   ```

## Detailed Workflow
1. **Speech Recognition:**
   Utilizes `transformers` pipeline with a pre-trained Whisper model for converting speech to text.

2. **Translation:**
   Employs a translation model to convert the recognized text from the source language to the target language.

3. **Text-to-Speech:**
   Uses a TTS model to convert the translated text back into speech, facilitating real-time communication.

## Contributing
Contributions are welcome! Please create a pull request or open an issue to discuss any changes or improvements.
