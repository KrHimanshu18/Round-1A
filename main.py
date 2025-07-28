# main.py
import os
import glob
from utils.gemini_trainer import GeminiTrainer
from utils.local_model import LocalHeadingModel
from utils.layout_utils import LayoutExtractor
from utils.postprocess import PostProcessor
import xgboost as xgb

# def run_phase1_generate_training_data():
#     """
#     Uses Gemini to create labeled training data from sample PDFs.
#     This phase requires an internet connection and a Gemini API key.
#     """
#     print("--- Starting Phase 1: Generating Training Data with Gemini ---")
    
#     # Ensure you have PDFs in this folder
#     training_pdf_paths = glob.glob("training_pdf/*.pdf")
#     if not training_pdf_paths:
#         print("Error: No PDFs found in the 'training_pdf' folder.")
#         print("Please add 1-3 sample PDFs to this folder to generate training data.")
#         return None

#     print(f"Found {len(training_pdf_paths)} PDFs for training data generation.")

#     # Initialize the Gemini trainer
#     # gemini = GeminiTrainer()
    
#     # This will use the Gemini API to label headings and save the results to a file inside the 'training_data' directory.
#     # training_data_file = gemini.create_training_data(training_pdf_paths)
#     training_data_file = "training_data/trained_output.json"
#     # training_data_file = "training_data/gemini_training_data.json"
    
#     print(f"--- Phase 1 Complete. Training data saved to: {training_data_file} ---")
#     return training_data_file

# def run_phase2_train_local_model(training_data_file: str):
#     """
#     Trains a local XGBoost model using the Gemini-generated data.
#     This phase is completely offline.
#     """
#     if not training_data_file or not os.path.exists(training_data_file):
#         print("Error: Training data file not found.")
#         print("Please run Phase 1 first to generate the training data.")
#         return False
        
#     print("\n--- Starting Phase 2: Training the Local Heading Model (XGBoost) ---")
    
#     local_model = LocalHeadingModel()
#     metrics = local_model.train_model(training_data_file)
    
#     if metrics:
#         print(f"Model trained successfully using XGBoost!")
#         print(f"Accuracy: {metrics.get('accuracy', 'N/A'):.2f}")
#         print("The trained model has been saved in the 'model/' directory.")
#         print("--- Phase 2 Complete ---")
#         return True
#     else:
#         print("Error: Model training failed.")
#         return False

def run_phase3_process_new_pdfs():
    """
    Uses the trained local model to find headings in new PDFs.
    This phase is completely offline and private.
    """
    print("\n--- Starting Phase 3: Processing New PDFs with Local Model ---")
    
    # Load the trained model
    local_model = LocalHeadingModel()
    if not local_model.load_model():
        print("Error: Failed to load the local model.")
        print("Please run Phase 2 to train the model first.")
        return

    # Find PDFs to process
    input_pdf = glob.glob("input_pdf/*.pdf")
    if not input_pdf:
        print("No new PDFs found in the 'input_pdf' folder.")
        print("Add the PDFs you want to analyze into that directory and run again.")
        return

    print(f"Found {len(input_pdf)} PDFs to process.")
    
    layout_extractor = LayoutExtractor()
    post_processor = PostProcessor()
    
    for pdf_path in input_pdf:
        print(f"\nProcessing: {os.path.basename(pdf_path)}...")
        
        # 1. Extract text blocks and layout information
        blocks = layout_extractor.extract_and_save_layout(pdf_path, "layout_data")
        if not blocks:
            print(f"Could not extract any text blocks from {pdf_path}. Skipping.")
            continue
            
        # 2. Use the local model to predict heading labels
        predictions = local_model.predict(blocks)
        
        # 3. Post-process the predictions into a clean JSON output
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        final_output = post_processor.process_predictions(blocks, predictions, pdf_name)
        
        output_path = os.path.join(post_processor.output_dir, f"{pdf_name}.json")
        print(f"Successfully processed. Results saved to: {output_path}")

    print("\n--- Phase 3 Complete ---")

if __name__ == '__main__':
    # # Step 1: Generate training data using a few sample PDFs
    # training_file = run_phase1_generate_training_data()
    
    # # Step 2: Train your local model using the generated data.
    # if training_file:
    #      run_phase2_train_local_model(training_file)

    # Step 3: Use your trained model to process new documents.
    run_phase3_process_new_pdfs()