import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class PDFReportWriter:
    def __init__(self, output_path="analysis_report.pdf"):
        self.output_path = output_path
        self.pdf = PdfPages(self.output_path)

    def save_figure(self):
        """Save the current matplotlib figure into the PDF."""
        fig = plt.gcf()
        self.pdf.savefig(fig)
        plt.close(fig)
        

    def save_table(self, df, title="Table Summary", fontsize=10):
        if df.empty:
            return

        # Reformat to Metric–Value style if it's a transposed summary table
        if df.shape[1] == 1:
            df = df.reset_index()
            df.columns = ['Metric', 'Value']

        fig_height = 0.5 * len(df) + 2  # Adjust height based on row count
        fig, ax = plt.subplots(figsize=(12, fig_height))
        ax.axis('off')

        # Add 1-line space by offsetting title vertically
        ax.set_title(f"{title}\n", fontsize=fontsize + 2, fontweight='bold', pad=5)

        table = ax.table(
            cellText=df.astype(str).values,
            colLabels=df.columns,
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        table.scale(1, 1.5)

        for i, key in table.get_celld().items():
            cell = table[i]
            cell.set_linewidth(0.5)
            if i[0] == 0:  # Header row
                cell.set_text_props(weight='bold')

        self.pdf.savefig(fig)
        plt.close(fig)



    def save_text(self, text, fontsize=12):
        """Save arbitrary text in the PDF."""
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.axis('off')
        ax.text(0.01, 0.5, text, ha='left', va='center', wrap=True, fontsize=fontsize)
        self.pdf.savefig(fig)
        plt.close(fig)

    def close(self):
        self.pdf.close()
