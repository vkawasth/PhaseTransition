import sys
import os
#sys.path.append('/Users/yanivschrader/Library/Python/3.9/bin/')
#sys.path.append('/Users/yanivschrader/.local/bin')


import spacy
from spacy_layout import spaCyLayout

# Make sure you load the specific model you downloaded
nlp = spacy.load("en_core_web_sm")
#nlp = spacy.blank("en")
layout = spaCyLayout(nlp)

# Process a document and create a spaCy Doc object
# Check if the filename is provided
if len(sys.argv) < 2:
    print("Usage: python read_file.py <filename>")
    sys.exit(1)

filename = sys.argv[1]

doc = layout(filename)

################### Process Core Text ####################################
# The text-based contents of the document
print("****** Entire Text *******")
print(doc.text)
print("****** Entire Text is printed *******")
doc_a = nlp(doc.text)
print("Noun phrases:", [chunk.text for chunk in doc_a.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc_a if token.pos_ == "VERB"])

print("Extracted Names (PERSON entities):")
for entity in doc_a.ents:
    #print(entity.text, entity.label_)
    if entity.label_ == "PERSON":
         print(f"NAME - {entity.text}")
################### Done with Core text ##################################


# Document layout including pages and page sizes
print(doc._.layout)
# Tables in the document and their extracted data
#print("DOC TABLES ******")
#print(doc._.tables)
#print("END DOC TABLES ******")
# Markdown representation of the document
#print(doc._.markdown)

# Layout spans for different sections
for span in doc.spans["layout"]:
    # Document section and token and character offsets into the text
    #print(span.text, span.start, span.end, span.start_char, span.end_char)
    # Section type, e.g. "text", "title", "section_header" etc.
    #print(span.label_)
    # Layout features of the section, including bounding box
    #print(span._.layout)
    # Closest heading to the span (accuracy depends on document structure)
    #print(span._.heading)
    doc = nlp(span.text)
    print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
    print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])
    print("SPAN Extracted Names (PERSON entities):")
    for entity in doc.ents:
        #print(entity.text, entity.label_)
        if entity.label_ == "PERSON":
            print(f"NAME - {entity.text}")


