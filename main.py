# main.py
import os
import glob
from utils.local_model import LocalHeadingModel
from utils.layout_utils import LayoutExtractor
from utils.postprocess import PostProcessor


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