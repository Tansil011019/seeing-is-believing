import pandas as pd
import xlsxwriter
import os

# --- CONFIGURATION ---
CSV_FILE = 'data/diagnoses.csv'
IMAGE_FOLDER = 'data/images'
OUTPUT_FILE = 'Expert_Annotation_Form.xlsx'
IMAGE_HEIGHT = 200  # Pixel height for images in Excel (increased for better fit)
LIKERT_SCALE = [1, 2, 3, 4, 5]

def create_expert_excel():
    # 1. Load the Data
    df = pd.read_csv(CSV_FILE)

    # 2. Initialize Excel Writer
    workbook = xlsxwriter.Workbook(OUTPUT_FILE)
    worksheet = workbook.add_worksheet("Annotations")

    # 3. Define Formats
    header_fmt = workbook.add_format({
        'bold': True, 'bg_color': '#D7E4BC', 'border': 1, 'align': 'center', 'valign': 'vcenter'
    })
    cell_fmt = workbook.add_format({
        'text_wrap': True, 'valign': 'top', 'border': 1
    })
    input_fmt = workbook.add_format({
        'bg_color': '#F2F2F2', 'border': 1, 'align': 'center', 'valign': 'vcenter'
    })

    # 4. Set Column Widths
    worksheet.set_column('A:A', 15)  # ID
    worksheet.set_column('B:C', 30)  # Label & Diagnosis (Text)
    worksheet.set_column('D:E', 25)  # Images columns
    worksheet.set_column('F:H', 15)  # Input columns (Likert)

    # 5. Write Headers
    headers = [
        "ID", "Label", "Diagnosis",
        "Original Image", "Heatmap Focus",
        "Heatmap\nRelevance (1-5)", "Diagnosis\nRelevance (1-5)", "Model\nTrust (1-5)"
    ]

    for col_num, header in enumerate(headers):
        worksheet.write(0, col_num, header, header_fmt)

    # 6. Iterate through data and fill rows
    for index, row in df.iterrows():
        row_num = index + 1  # Start from row 1 (row 0 is header)

        # Set row height to accommodate images
        worksheet.set_row(row_num, IMAGE_HEIGHT)

        # Write Text Data
        worksheet.write(row_num, 0, row['id'], cell_fmt)
        worksheet.write(row_num, 1, row['label'], cell_fmt)
        worksheet.write(row_num, 2, row['diagnosis'], cell_fmt)

        # Write Data Validation (Dropdowns) for inputs
        # Use data_validation to force input to be 1-5
        for col_idx in range(5, 8):  # Columns F, G, H
            worksheet.data_validation(row_num, col_idx, row_num, col_idx, {
                'validate': 'list',
                'source': LIKERT_SCALE,
                'input_title': 'Rate 1-5',
                'input_message': 'Select a value from the list'
            })
            # Write a blank initial value formatted with color
            worksheet.write(row_num, col_idx, "", input_fmt)

        # 7. Insert Images
        # Function to helper insert image with scaling
        def insert_img(filename, col_idx):
            img_path = os.path.join(IMAGE_FOLDER, filename)

            if os.path.exists(img_path):
                # Insert image, scaling it to fit the cell roughly
                # Note: xlsxwriter inserts based on offsets.
                # We use object_position=1 to move and size with cells
                worksheet.insert_image(row_num, col_idx, img_path, {
                    'x_scale': 0.15, 'y_scale': 0.15,  # Adjusted scale for better cell fit
                    'object_position': 1,
                    'y_offset': 5, 'x_offset': 5
                })
            else:
                worksheet.write(row_num, col_idx, "Image Not Found", cell_fmt)

        # Insert Original
        insert_img(f"{row['id']}.jpg", 3)
        # Insert Heatmap
        insert_img(f"heatmap_{row['id']}.jpg", 4)

    workbook.close()
    print(f"Successfully created {OUTPUT_FILE}")

if __name__ == "__main__":
    create_expert_excel()
