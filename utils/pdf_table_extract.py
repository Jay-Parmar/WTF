import camelot
import pdfplumber
import pandas as pd

def extract_tables_with_camelot(pdf_path, pages='all', flavor='lattice'):
    try:
        tables = camelot.read_pdf(pdf_path, pages=pages, flavor=flavor)
        return [table.df for table in tables]
    except Exception as e:
        print(f"Camelot extraction failed: {e}")
        return []

def extract_tables_with_pdfplumber(pdf_path, max_tables=5):
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                for tbl in page_tables:
                    tables.append(pd.DataFrame(tbl))
                    if len(tables) >= max_tables:
                        return tables
    except Exception as e:
        print(f"pdfplumber extraction failed: {e}")
    return tables

def extract_tables_from_pdf(pdf_path, pages='all', flavor='lattice'):
    tables = extract_tables_with_camelot(pdf_path, pages, flavor)
    if not tables:
        print("Falling back to pdfplumber...")
        tables = extract_tables_with_pdfplumber(pdf_path)
    return tables

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_table_extract.py yourfile.pdf")
    else:
        pdf_path = sys.argv[1]
        tables = extract_tables_from_pdf(pdf_path)
        print(f"Found {len(tables)} tables.")
        for idx, df in enumerate(tables):
            print(f"\nTable {idx+1}:\n")
            print(df.head())
