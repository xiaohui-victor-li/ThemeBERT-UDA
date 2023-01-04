# NLP PDF to Text Module
# ------------------------
#
# apply the @pdfminer library
#
# https://pdfminersix.readthedocs.io/en/latest/index.html
#
from pdfminer.high_level import extract_text

from io import StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

class PDFprocessor:
    """PDF to Text Processor

    Core Function:
        @process
    """
    def __init__(self, line_overlap=0.5, char_margin=2.0, line_margin=0.5, word_margin=0.1, boxes_flow=0.5):
        """
        :param line_overlap: If two characters have more overlap than this they 
        are considered to be on the same line.
        :param char_margin: If two characters are closer together than this margin
        they are considered part of the same line.
        :param boxes_flow: Specifies how much a horizontal and vertical position of a text
        matters when determining the order of text boxes.
        """

        self.param_config = LAParams(line_overlap=line_overlap,
                                     char_margin=char_margin,
                                     line_margin=line_margin,
                                     word_margin=word_margin,
                                     boxes_flow=boxes_flow
                                    )
        self.page_number = 0
        
    def process(self, doc_file, page_ids=None):
        """Parse and process the PDF into text data by pages
        :param doc_file: file directory
        :param pages: a list of page to parse, by default all the pages (None)

        Output:
            the PDF output obj
        """
        self.output_dict = {}
        with open(doc_file, 'rb') as in_file:
            self.parser = PDFParser(in_file)
            self.doc = PDFDocument(self.parser)
            self.rsrcmgr = PDFResourceManager()

            ## iterate across pages
            for i,page in enumerate(PDFPage.create_pages(self.doc)):
                if (page_ids is None) or (i in page_ids):
                    output = StringIO() 
                    self.device = TextConverter(self.rsrcmgr, output, laparams=self.param_config)

                    self.interpreter = PDFPageInterpreter(self.rsrcmgr, self.device)
                    self.interpreter.process_page(page)

                    self.output_dict[i] = output
                    self.page_number += 1

        return self.output_dict