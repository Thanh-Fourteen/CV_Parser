from Pipline.utils4zero2hero import *

# Tắt cảnh báo từ transformers
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    pipeline = CVPipeline(
        image_input_dir=os.path.join(os.getcwd(), "100_file_testing", "Arts resumes"),
        text_output_dir=os.path.join(os.getcwd(), "data_extracted_output", "text_extracted"),
        summary_output_dir=os.path.join(os.getcwd(), "data_extracted_output", "summaries"),
        db_path=os.path.join(os.getcwd(), "data_extracted_output", "CV_parser_database.db"),
        model_path = os.path.join(os.getcwd(), "db_167_1.9499_1.9947.h5")
    )
    
    job_desc = """
    SKILLS
    Staff Training, schedule, Experience with Medical Records, General Knowledge Of Computer
    Software, On Time And Reliable, Weekend Availability, Works Well As Part OF A Team Or
    Individually, Excellent Multi-tasker, Rapid Order Processing, Conflict Resolution Techniques,
    Results-oriented, Marketing, And Advertising,
    """
    top_candidate = 1
    
    pipeline.run(job_desc, top_candidate)