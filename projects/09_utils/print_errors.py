# looks at the mismatches in the output
# of prediction_analyser and prints them
# to an Excel sheet

import argparse
import xlsxwriter
import sys
import codecs


if sys.version_info[0] >= 3:
    unicode = str


def to_unicode(text, errors='strict', encoding='utf8'):
    """Convert a string (unicode or bytestring in `encoding`),
    to bytestring in utf8."""
    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode('utf8')

# file to analyse
parser = argparse.ArgumentParser(
    description="Analyse the entities of the file")
parser.add_argument("--read_file",
                    help="file to be analysed",
                    required=True)
parser.add_argument("--write_file",
                    help="Excel sheet name",
                    required=True)
args = parser.parse_args()

f = codecs.open(args.read_file, "r", "utf-8")
contents = f.read()
contents = contents.split("\n")

ref = contents[1].split()[2]
entity_type = contents[1].split()[5]

book = xlsxwriter.Workbook(args.write_file)
sheet = book.add_worksheet(entity_type + "_" + ref)
sheet.set_column('C:C', 20)
sheet.set_column('D:E', 20)

# add style for multi line text in a cell
bold = book.add_format({'bold': True, 'font_size': 13})
unicode_font = book.add_format({'text_wrap': True, 'font_name': "Menlo", 'font_size': 13})

row_idx = 1
col_idx = 1

sheet.write(row_idx, col_idx, 'Line', bold)
sheet.write(row_idx, col_idx+1, 'Word', bold)
sheet.write(row_idx, col_idx+2, 'True', bold)
sheet.write(row_idx, col_idx+3, 'Predict', bold)
row_idx += 1

words = []
line_nos = []
predict_tags = []
true_tags = []

line = 7
idx = 1
for l in contents[6:]:
    if l != "":
        line_nos.append(l.split()[0])
        words.append(l.split()[1])
        true_tags.append(l.split()[2])
        predict_tags.append(l.split()[3])
    else:
        if true_tags == predict_tags:
            line_nos = []
            words = []
            true_tags = []
            predict_tags = []
        else:
            sheet.write(row_idx, col_idx-1, idx, unicode_font)
            sheet.write(row_idx, col_idx, "\n".join(line_nos), unicode_font)
            sheet.write(row_idx, col_idx+1, "\n".join(words), unicode_font)
            sheet.write(row_idx, col_idx+2, "\n".join(true_tags), unicode_font)
            sheet.write(row_idx, col_idx+3, "\n".join(predict_tags), unicode_font)
            row_idx += 1
            idx += 1
            line_nos = []
            words = []
            true_tags = []
            predict_tags = []

book.close()
