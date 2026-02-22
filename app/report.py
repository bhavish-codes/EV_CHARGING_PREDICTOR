from fpdf import FPDF
import datetime

class EVReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'EV Charging Infrastructure Planning Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(demand_summary, recommendations, filename="ev_planning_report.pdf"):
    pdf = EVReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Report Info
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.ln(5)
    
    # Demand Summary
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "1. Charging Demand Summary", 0, 1)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, demand_summary)
    pdf.ln(5)
    
    # Recommendations
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. Infrastructure & Scheduling Recommendations", 0, 1)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, recommendations)
    pdf.ln(10)
    
    # References
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, "Supporting references: UrbanEV Dataset & Shenzhen EV Planning Guidelines.", 0, 1)
    
    pdf.output(filename)
    return filename

if __name__ == "__main__":
    generate_pdf_report("High demand at Peak", "Add 5 chargers", "test_report.pdf")
    print("Test report generated.")
