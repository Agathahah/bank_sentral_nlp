# -*- coding: utf-8 -*-
"""
utils.py - Utility functions for Bank Sentral NLP project
"""

import os
import torch
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle

def setup_device():
    """
    Setup and return the appropriate device (CUDA, MPS, or CPU)
    
    Returns:
    --------
    torch.device
        The appropriate device for computation
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device

def configure_logging(log_file=None, level=logging.INFO):
    """
    Configure logging for the project
    
    Parameters:
    -----------
    log_file : str, optional
        Path to log file
    level : int, default logging.INFO
        Logging level
    """
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Setup file handler if log_file is provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def load_data(file_path):
    """
    Load data from file (Excel or CSV)
    
    Parameters:
    -----------
    file_path : str
        Path to data file
        
    Returns:
    --------
    pd.DataFrame
        Loaded data
    """
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        print(f"Loaded data from {file_path}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def save_results(df, output_path, create_dir=True):
    """
    Save DataFrame to file
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to save
    output_path : str
        Path to output file
    create_dir : bool, default True
        Whether to create directory if it doesn't exist
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        if create_dir:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if output_path.endswith('.xlsx'):
            df.to_excel(output_path, index=False)
        elif output_path.endswith('.csv'):
            df.to_excel(output_path, index=False)
        elif output_path.endswith('.pkl'):
            with open(output_path, 'wb') as f:
                pickle.dump(df, f)
        else:
            raise ValueError(f"Unsupported file format: {output_path}")
        
        print(f"Results saved to {output_path}")
        return True
    
    except Exception as e:
        print(f"Error saving results: {e}")
        return False

def plot_confusion_matrix(cm, labels, output_path=None, title='Confusion Matrix', figsize=(10, 8)):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    cm : numpy.ndarray
        Confusion matrix
    labels : list
        Class labels
    output_path : str, optional
        Path to save the plot
    title : str, default 'Confusion Matrix'
        Plot title
    figsize : tuple, default (10, 8)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure
    """
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    return plt.gcf()

def generate_report_summary(metrics, output_path=None):
    """
    Generate and save performance report summary
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing performance metrics
    output_path : str, optional
        Path to save the report
        
    Returns:
    --------
    str
        Report summary text
    """
    report = f"Performance Report\n{'='*50}\n"
    report += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report += "Metrics Summary:\n"
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            report += f"- {metric}: {value:.4f}\n"
        else:
            report += f"- {metric}: {value}\n"
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {output_path}")
    
    return report

def extract_embeddings(model, tokenizer, texts, batch_size=4, max_length=512, device=None):
    """
    Extract embeddings from texts using IndoBERT
    
    Parameters:
    -----------
    model : transformers.PreTrainedModel
        IndoBERT model
    tokenizer : transformers.PreTrainedTokenizer
        IndoBERT tokenizer
    texts : list
        List of texts
    batch_size : int, default 4
        Batch size
    max_length : int, default 512
        Maximum sequence length
    device : torch.device, optional
        Device to use for computation
        
    Returns:
    --------
    numpy.ndarray
        Text embeddings
    """
    if device is None:
        device = setup_device()
    
    model.to(device)
    model.eval()
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the [CLS] token representation as document embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        all_embeddings.append(embeddings)
    
    return np.vstack(all_embeddings)